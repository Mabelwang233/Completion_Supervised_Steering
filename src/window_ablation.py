"""
Response-length ablation for completion-RFM -- domain-agnostic version
of window_ablation_personas.py, works for both Personas and Fears via
--domain {persona, fear}. All the windowing/training logic
(completion_rfm_windowed.py, make_eval_subset.py) was already
domain-agnostic; only the concept-dir naming, manifest concept-name
key, and steer/judge modules (handled inside eval_one_config.py)
differ between the two domains -- see DOMAIN_CONFIG.

Configs (11 total, same as the persona version):
  fixed:      0, 15, 30, 60, full          (0 == vanilla RFM, full == unwindowed completion-RFM)
  anchor_frac: 0%, 10%, 25%, 50%, 75%, 100% (0% == vanilla RFM, 100% == unwindowed completion-RFM)
deduplicated to 9 distinct method labels (7 newly trained + 2 shared/reused).

Directory layout expected per concept (matching your existing runs):
    <concept_dir>/baseline_directions.pkl
    <concept_dir>/<strategy>/dataset_with_completions.json
    <concept_dir>/<strategy>/completion_rfm_directions.pkl

Outputs go to <concept_dir>/<strategy>/window_ablation/.

NOTE ON CSV SCHEMA: per_example_results.csv now has a "domain" column
and a generic "concept" column (instead of a domain-specific
"persona"/"fear" column), so one schema covers both domains. This is a
breaking change vs. the persona-only per_example_results.csv you
already have from the earlier run -- if you want fear results in a
SEPARATE file (recommended, since the columns differ), point
--output_dir at a new fears output directory rather than reusing the
personas one; if you re-run personas with this version later, do it
into a fresh window_ablation/ dir too rather than appending to the old one.
"""
import argparse
import csv
import gc
import json
import os
import pickle

import torch

from utils import load_model
from completion_rfm_windowed import train_completion_rfm_windowed
from make_eval_subset import make_eval_subset
from eval_one_config import evaluate_one_config

FIXED_TOKEN_WINDOWS = [15, 30, 60]              # 0 and "full" handled separately (shared baselines)
ANCHOR_FRAC_WINDOWS = [0.10, 0.25, 0.50, 0.75]  # 0.0 and 1.0 handled separately

DOMAIN_CONFIG = {
    "persona": {"manifest_key": "persona", "dir_prefix": "persona_"},
    "fear": {"manifest_key": "fear", "dir_prefix": "fear_"},
}


def load_directions_pkl(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["directions_per_layer"]


def run_ablation(concept_dir, domain, strategy="long_subsampled_anchor",
                  model_name="meta-llama/Llama-3.1-8B-Instruct",
                  alpha_grid=(0.3, 0.4, 0.5, 0.6, 0.7), n_eval=20, eval_seed=0,
                  judge_model="gpt-4o-mini", eval_gen_tokens=100,
                  cache_dir="/data/mabel/cache/MCL_rebuttal/Data_Models/hf_cache",
                  rfm_iters=8, train_seed=0, language_model=None, tokenizer=None):
    """
    language_model/tokenizer: pass an already-loaded model to reuse it
    across multiple concepts (see run_all_concepts below). If omitted,
    this loads its own copy and does NOT free it afterward -- fine for
    a single concept via the CLI, but don't call this in a loop without
    passing a shared model in, or you'll hit the OOM from before.
    """
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"unknown domain: {domain!r} -- must be one of {list(DOMAIN_CONFIG)}")

    strategy_dir = os.path.join(concept_dir, strategy)
    dataset_path = os.path.join(strategy_dir, "dataset_with_completions.json")
    baseline_path = os.path.join(concept_dir, "baseline_directions.pkl")
    full_completion_rfm_path = os.path.join(strategy_dir, "completion_rfm_directions.pkl")

    for required in [dataset_path, baseline_path, full_completion_rfm_path]:
        assert os.path.exists(required), f"missing required file: {required}"

    out_dir = os.path.join(strategy_dir, "window_ablation")
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. fixed eval subset (build once, reuse if it already exists) ----
    eval_subset_path = os.path.join(out_dir, "eval_subset.json")
    if not os.path.exists(eval_subset_path):
        make_eval_subset(dataset_path, eval_subset_path, n=n_eval, seed=eval_seed)
    else:
        print(f"{eval_subset_path} exists, reusing.")

    # ---- 2. train only the configs that don't already exist ----
    configs = []
    for w in FIXED_TOKEN_WINDOWS:
        path = os.path.join(out_dir, f"completion_rfm_w{w}.pkl")
        configs.append((f"completion_rfm_fixed{w}", path, "fixed", {"window_tokens": w}))
    for f in ANCHOR_FRAC_WINDOWS:
        tag = f"{int(round(f * 100)):03d}"
        path = os.path.join(out_dir, f"completion_rfm_af{tag}.pkl")
        configs.append((f"completion_rfm_anchorfrac{tag}", path, "anchor_frac", {"window_frac": f}))

    owns_model = language_model is None
    if owns_model:
        print("\nLoading model...")
        language_model, tokenizer = load_model(model_name, cache_dir=cache_dir)

    for method_label, path, window_type, kwargs in configs:
        if os.path.exists(path):
            print(f"{path} exists, skipping training.")
            continue
        print(f"\nTraining {method_label} ...")
        train_completion_rfm_windowed(
            dataset_path, path, model_name, window_type,
            window_tokens=kwargs.get("window_tokens"), window_frac=kwargs.get("window_frac"),
            rfm_iters=rfm_iters, cache_dir=cache_dir, seed=train_seed,
            language_model=language_model, tokenizer=tokenizer,
        )

    # ---- 3. evaluate every config (including the 2 shared baselines) on the fixed subset ----
    language_model.eval()

    eval_targets = [("completion_rfm_fixed0_vanilla", baseline_path, "rfm_key")] + \
                   [(label, path, "direct") for label, path, _, _ in configs] + \
                   [("completion_rfm_fixed_full", full_completion_rfm_path, "direct"),
                    ("completion_rfm_anchorfrac100", full_completion_rfm_path, "direct")]

    all_summaries = []
    for method_label, path, load_mode in eval_targets:
        if load_mode == "rfm_key":
            with open(path, "rb") as f:
                baselines = pickle.load(f)
            directions_per_layer = baselines["rfm"]["directions_per_layer"]
        else:
            directions_per_layer = load_directions_pkl(path)

        summary = evaluate_one_config(
            language_model, tokenizer, dataset_path, eval_subset_path,
            method_label, directions_per_layer, list(alpha_grid), out_dir, domain,
            judge_model=judge_model, max_new_tokens=eval_gen_tokens, seed=train_seed,
        )
        all_summaries.append(summary)

    # ---- 4. combined summary for this concept ----
    combined_path = os.path.join(out_dir, "combined_summary.csv")
    with open(combined_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "concept", "method", "best_alpha", "best_mean_score", "n_eval"])
        for s in all_summaries:
            writer.writerow([s["domain"], s["concept"], s["method"], s["best_alpha"],
                              f"{s['best_mean_score']:.3f}", s["n_eval"]])
    print(f"\nWrote {combined_path}")


def run_all_concepts(manifest_path, output_dir, domain, strategy="long_subsampled_anchor",
                      model_name="meta-llama/Llama-3.1-8B-Instruct",
                      alpha_grid=(0.3, 0.4, 0.5, 0.6, 0.7), n_eval=20, eval_seed=0,
                      judge_model="gpt-4o-mini", eval_gen_tokens=100,
                      cache_dir="/data/mabel/cache/MCL_rebuttal/Data_Models/hf_cache",
                      rfm_iters=8, train_seed=0):
    """
    Loops run_ablation over every concept in manifest.json, loading the
    model ONCE and sharing it across all concepts (looping run_ablation
    directly without this would reload the model per concept and
    reproduce the earlier OOM).
    """
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"unknown domain: {domain!r} -- must be one of {list(DOMAIN_CONFIG)}")
    manifest_key = DOMAIN_CONFIG[domain]["manifest_key"]
    dir_prefix = DOMAIN_CONFIG[domain]["dir_prefix"]

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Loading model once, shared across {len(manifest)} {domain} concepts...")
    language_model, tokenizer = load_model(model_name, cache_dir=cache_dir)
    language_model.eval()

    combined_rows = []
    for entry in manifest:
        concept_name = entry[manifest_key]
        safe_name = concept_name.lower().replace(" ", "_")
        concept_dir = os.path.join(output_dir, f"{dir_prefix}{safe_name}")

        print(f"\n{'=' * 60}\n{domain.upper()}: {concept_name}\n{'=' * 60}")
        try:
            run_ablation(
                concept_dir, domain, strategy, model_name, alpha_grid, n_eval, eval_seed,
                judge_model, eval_gen_tokens, cache_dir, rfm_iters, train_seed,
                language_model=language_model, tokenizer=tokenizer,
            )
        except AssertionError as e:
            print(f"  SKIPPING {concept_name}: {e}")
            continue

        per_concept_summary = os.path.join(concept_dir, strategy, "window_ablation", "combined_summary.csv")
        if os.path.exists(per_concept_summary):
            with open(per_concept_summary) as f:
                reader = csv.reader(f)
                next(reader)  # header
                combined_rows.extend(reader)

        torch.cuda.empty_cache()

    all_path = os.path.join(output_dir, f"window_ablation_all_{domain}s_summary.csv")
    with open(all_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "concept", "method", "best_alpha", "best_mean_score", "n_eval"])
        writer.writerows(combined_rows)
    print(f"\nAll {domain}s done. Combined summary: {all_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=["persona", "fear"])
    p.add_argument("--concept_dir", default=None,
                    help="single-concept mode, e.g. .../fears_llama_8b_alllayers/fear_heights")
    p.add_argument("--manifest_path", default=None,
                    help="all-concepts mode: path to manifest.json")
    p.add_argument("--output_dir", default=None,
                    help="all-concepts mode: e.g. .../outputs/fears_llama_8b_alllayers "
                         "(each concept dir derived as <output_dir>/<persona_ or fear_><safe_name>)")
    p.add_argument("--strategy", default="long_subsampled_anchor")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--alpha_grid", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    p.add_argument("--n_eval", type=int, default=20)
    p.add_argument("--eval_seed", type=int, default=0)
    p.add_argument("--judge_model", default="gpt-4o-mini")
    p.add_argument("--eval_gen_tokens", type=int, default=100)
    p.add_argument("--cache_dir", default="/data/mabel/cache/MCL_rebuttal/Data_Models/hf_cache")
    p.add_argument("--rfm_iters", type=int, default=8)
    p.add_argument("--train_seed", type=int, default=0)
    args = p.parse_args()

    if args.manifest_path and args.output_dir:
        run_all_concepts(args.manifest_path, args.output_dir, args.domain, args.strategy, args.model,
                          args.alpha_grid, args.n_eval, args.eval_seed, args.judge_model,
                          args.eval_gen_tokens, args.cache_dir, args.rfm_iters, args.train_seed)
    elif args.concept_dir:
        run_ablation(args.concept_dir, args.domain, args.strategy, args.model, args.alpha_grid,
                     args.n_eval, args.eval_seed, args.judge_model, args.eval_gen_tokens,
                     args.cache_dir, args.rfm_iters, args.train_seed)
    else:
        p.error("pass either --concept_dir (single concept) or both --manifest_path and "
                "--output_dir (all concepts)")
