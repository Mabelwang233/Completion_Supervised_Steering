"""
End-to-end attention-guided completion-RFM: compute tau*, train
per-layer directions, evaluate on the SAME fixed 20-question subset
window_ablation.py already used for this concept (copied over if it
exists, so this method's numbers are directly comparable to the
window-length sweep results you already have).
"""
import argparse
import csv
import json
import os
import pickle
import shutil

import torch

from utils import load_model
from attention_guided_tau import compute_tau_star_dataset
from attention_guided_completion_rfm import train_attention_guided_completion_rfm, ATTENTION_LAYERS_DEFAULT
from make_eval_subset import make_eval_subset
from eval_one_config import evaluate_one_config

DOMAIN_CONFIG = {
    "persona": {"manifest_key": "persona", "dir_prefix": "persona_"},
    "fear": {"manifest_key": "fear", "dir_prefix": "fear_"},
}


def run_one_concept(concept_dir, domain, strategy="long_subsampled_anchor",
                     model_name="meta-llama/Llama-3.1-8B-Instruct",
                     alpha_grid=(0.3, 0.4, 0.5, 0.6, 0.7), n_eval=20, eval_seed=0,
                     judge_model="gpt-4o-mini", eval_gen_tokens=100,
                     cache_dir="/data/mabel/cache/MCL_rebuttal/Data_Models/hf_cache",
                     rfm_iters=8, layers=None, seed=0,
                     language_model=None, tokenizer=None):
    strategy_dir = os.path.join(concept_dir, strategy)
    dataset_path = os.path.join(strategy_dir, "dataset_with_completions.json")
    assert os.path.exists(dataset_path), f"missing: {dataset_path}"

    out_dir = os.path.join(strategy_dir, "attention_guided")
    os.makedirs(out_dir, exist_ok=True)

    # reuse window_ablation's eval subset if it exists AND matches n_eval, so this stays directly
    # comparable to the window-length sweep; otherwise build (or rebuild) a fresh one at n_eval
    window_eval_subset = os.path.join(strategy_dir, "window_ablation", "eval_subset.json")
    eval_subset_path = os.path.join(out_dir, "eval_subset.json")

    def _subset_n(path):
        with open(path) as f:
            return json.load(f)["n"]

    if os.path.exists(eval_subset_path):
        existing_n = _subset_n(eval_subset_path)
        if existing_n != n_eval:
            print(f"WARNING: {eval_subset_path} exists with n={existing_n}, but n_eval={n_eval} was "
                  f"requested -- regenerating (breaks pairing with any prior n={existing_n} run).")
            make_eval_subset(dataset_path, eval_subset_path, n=n_eval, seed=eval_seed)
        else:
            print(f"{eval_subset_path} exists with matching n={n_eval}, reusing.")
    elif os.path.exists(window_eval_subset) and _subset_n(window_eval_subset) == n_eval:
        shutil.copy(window_eval_subset, eval_subset_path)
        print(f"Reusing window_ablation's eval subset (n={n_eval}): {window_eval_subset}")
    else:
        make_eval_subset(dataset_path, eval_subset_path, n=n_eval, seed=eval_seed)

    owns_model = language_model is None
    if owns_model:
        language_model, tokenizer = load_model(model_name, cache_dir=cache_dir)
    language_model.eval()

    tau_star_path = os.path.join(out_dir, "tau_star.json")
    if not os.path.exists(tau_star_path):
        compute_tau_star_dataset(dataset_path, domain, model_name, tau_star_path,
                                  layers=layers, cache_dir=cache_dir,
                                  language_model=language_model, tokenizer=tokenizer)
    else:
        print(f"{tau_star_path} exists, reusing.")

    directions_path = os.path.join(out_dir, "attention_guided_completion_rfm_directions.pkl")
    if not os.path.exists(directions_path):
        train_attention_guided_completion_rfm(
            dataset_path, tau_star_path, directions_path, model_name, layers=layers,
            rfm_iters=rfm_iters, cache_dir=cache_dir, seed=seed,
            language_model=language_model, tokenizer=tokenizer,
        )
    else:
        print(f"{directions_path} exists, reusing.")

    with open(directions_path, "rb") as f:
        directions_per_layer = pickle.load(f)["directions_per_layer"]

    summary = evaluate_one_config(
        language_model, tokenizer, dataset_path, eval_subset_path,
        "attention_guided_completion_rfm", directions_per_layer, list(alpha_grid), out_dir, domain,
        judge_model=judge_model, max_new_tokens=eval_gen_tokens, seed=seed,
    )

    combined_path = os.path.join(out_dir, "combined_summary.csv")
    with open(combined_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "concept", "method", "best_alpha", "best_mean_score", "n_eval"])
        writer.writerow([summary["domain"], summary["concept"], summary["method"],
                          summary["best_alpha"], f"{summary['best_mean_score']:.3f}", summary["n_eval"]])
    print(f"Wrote {combined_path}")
    return summary


def run_all_concepts(manifest_path, output_dir, domain, strategy="long_subsampled_anchor",
                      model_name="meta-llama/Llama-3.1-8B-Instruct",
                      alpha_grid=(0.3, 0.4, 0.5, 0.6, 0.7), n_eval=20, eval_seed=0,
                      judge_model="gpt-4o-mini", eval_gen_tokens=100,
                      cache_dir="/data/mabel/cache/MCL_rebuttal/Data_Models/hf_cache",
                      rfm_iters=8, layers=None, seed=0):
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"unknown domain: {domain!r}")
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
            run_one_concept(
                concept_dir, domain, strategy, model_name, alpha_grid, n_eval, eval_seed,
                judge_model, eval_gen_tokens, cache_dir, rfm_iters, layers, seed,
                language_model=language_model, tokenizer=tokenizer,
            )
        except (AssertionError, NotImplementedError) as e:
            print(f"  SKIPPING {concept_name}: {e}")
            continue

        per_concept_summary = os.path.join(concept_dir, strategy, "attention_guided", "combined_summary.csv")
        if os.path.exists(per_concept_summary):
            with open(per_concept_summary) as f:
                reader = csv.reader(f)
                next(reader)
                combined_rows.extend(reader)

        torch.cuda.empty_cache()

    all_path = os.path.join(output_dir, f"attention_guided_all_{domain}s_summary.csv")
    with open(all_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "concept", "method", "best_alpha", "best_mean_score", "n_eval"])
        writer.writerows(combined_rows)
    print(f"\nAll {domain}s done. Combined summary: {all_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=["persona", "fear"])
    p.add_argument("--concept_dir", default=None, help="single-concept mode")
    p.add_argument("--manifest_path", default=None, help="all-concepts mode")
    p.add_argument("--output_dir", default=None, help="all-concepts mode")
    p.add_argument("--strategy", default="long_subsampled_anchor")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--alpha_grid", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    p.add_argument("--n_eval", type=int, default=20)
    p.add_argument("--eval_seed", type=int, default=0)
    p.add_argument("--judge_model", default="gpt-4o-mini")
    p.add_argument("--eval_gen_tokens", type=int, default=100)
    p.add_argument("--cache_dir", default="/data/mabel/cache/MCL_rebuttal/Data_Models/hf_cache")
    p.add_argument("--rfm_iters", type=int, default=8)
    p.add_argument("--layers", nargs="+", type=int, default=None,
                    help="defaults to ATTENTION_LAYERS_DEFAULT in attention_guided_completion_rfm.py (-8..-24)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.manifest_path and args.output_dir:
        run_all_concepts(args.manifest_path, args.output_dir, args.domain, args.strategy, args.model,
                          args.alpha_grid, args.n_eval, args.eval_seed, args.judge_model,
                          args.eval_gen_tokens, args.cache_dir, args.rfm_iters, args.layers, args.seed)
    elif args.concept_dir:
        run_one_concept(args.concept_dir, args.domain, args.strategy, args.model, args.alpha_grid,
                        args.n_eval, args.eval_seed, args.judge_model, args.eval_gen_tokens,
                        args.cache_dir, args.rfm_iters, args.layers, args.seed)
    else:
        p.error("pass either --concept_dir (single concept) or both --manifest_path and --output_dir (all concepts)")
