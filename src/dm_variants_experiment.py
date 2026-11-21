"""
Diff-means token-pooling ablation: the two missing cells of the 2x2

                     last token                    mean over all tokens
    prompt x         P-DM   (baselines.py)         prompt_dm_all          <- new
    completion z     completion_dm_lastonly <- new C-DM (bag_of_words_diff_means.py)

  prompt_dm_all          : raw prompt (no chat template, exactly what baselines.py
                           feeds P-DM), hidden states mean-pooled over every
                           non-pad prompt token except the first (position 0
                           carries attention-sink activations), then diff-means.
  completion_dm_lastonly : last token of formatted_prompt + completion (the
                           final completion token), diff-means over labels --
                           identical recipe to completion_diff_means.py, reusing
                           the same dataset_with_completions.json as C-DM.

Directions are unit-normalized per layer and use the same layer indexing
(-1 .. -(L-1)) as direction_utils.get_hidden_states, so they plug into the
existing {persona,fear,expert}_steer_and_evaluate.py via baseline_path.

One method per invocation (--method), in two stages:
  --stage directions : one model load, directions for every concept
  --stage eval       : alpha sweep + judge per concept (long)

Outputs:
  <out_dir>/<category>s/<category>_<slug>/<method>/directions.pkl
  <out_dir>/<category>s/<category>_<slug>/<method>/{summary,per_example_results}.csv
  <out_dir>/combined_summary_<method>_<category>.csv
"""
import argparse
import csv
import json
import os
import pickle
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_model
from neural_controllers import NeuralController

CATEGORIES = ("persona", "fear", "expert")
METHODS = ("completion_dm_lastonly", "prompt_dm_all")
STRATEGY = "long_subsampled_anchor"


def concept_list(data_dir, self_out_dir, model_tag, category):
    with open(os.path.join(data_dir, f"{category}s", "manifest30.json")) as f:
        manifest = json.load(f)
    rows = []
    for entry in manifest:
        name = entry[category]
        slug = name.lower().replace(" ", "_")
        completions = os.path.join(self_out_dir, f"{category}s_{model_tag}_alllayers",
                                   f"{category}_{slug}", STRATEGY, "dataset_with_completions.json")
        rows.append({"name": name, "slug": slug, "dataset": entry["path"], "completions": completions})
    return rows


def prompt_dm_all(language_model, tokenizer, prompts, labels, batch_size=8):
    """Mean over all non-pad prompt tokens except the first, then diff-means.
    Returns {layer: (d,)}."""
    num_layers = len(language_model.model.layers)
    layers = list(range(-1, -num_layers, -1))
    pooled = {layer: [] for layer in layers}
    with torch.no_grad():
        for start in tqdm(range(0, len(prompts), batch_size), desc="prompt_dm_all"):
            enc = tokenizer(prompts[start:start + batch_size], return_tensors="pt",
                            padding=True, add_special_tokens=False).to(language_model.device)
            out = language_model(**enc, output_hidden_states=True)
            mask = enc["attention_mask"].clone()
            # drop each prompt's first real token (left padding -> first nonzero mask entry)
            first = mask.argmax(dim=1)
            mask[torch.arange(mask.shape[0]), first] = 0
            mask = mask.unsqueeze(-1).float()  # (B, T, 1)
            # same layer <-> hidden_state pairing as direction_utils.get_hidden_states
            for layer, hidden in zip(layers, reversed(out.hidden_states)):
                mean = (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1)
                pooled[layer].append(mean.cpu())
    labels = torch.tensor(labels)
    directions = {}
    for layer in layers:
        x = torch.cat(pooled[layer], dim=0)
        direction = x[labels == 1].mean(dim=0) - x[labels == 0].mean(dim=0)
        directions[layer] = direction / direction.norm()
    return directions, layers


def completion_dm_lastonly(language_model, tokenizer, completions_path, batch_size=8):
    """Same as completion_diff_means.train_completion_diff_means, with a preloaded model."""
    with open(completions_path) as f:
        train = json.load(f)["train"]
    full_texts = [ex["formatted_prompt"] + ex["completion"] for ex in train]
    labels = [ex["label"] for ex in train]
    controller = NeuralController(language_model, tokenizer,
                                  control_method="mean_difference", batch_size=batch_size)
    controller.compute_directions(full_texts, labels)
    directions = {layer: comps[0].detach().cpu() for layer, comps in controller.directions.items()}
    return directions, controller.hidden_layers


def method_dir(args, category, concept):
    return os.path.join(args.out_dir, f"{category}s", f"{category}_{concept['slug']}", args.method)


def run_directions(concepts_by_category, args):
    language_model, tokenizer = load_model(args.model, cache_dir=args.cache_dir)
    language_model.eval()
    for category, concepts in concepts_by_category.items():
        for c in concepts:
            out_path = os.path.join(method_dir(args, category, c), "directions.pkl")
            if os.path.exists(out_path):
                print(f"  {out_path} exists, skipping.")
                continue
            print(f"\n=== {args.method} | {category}: {c['name']} ===")
            if args.method == "prompt_dm_all":
                with open(c["dataset"]) as f:
                    train = json.load(f)["train"]
                directions, layers = prompt_dm_all(
                    language_model, tokenizer, [ex["prompt"] for ex in train],
                    [ex["label"] for ex in train], args.batch_size)
            else:
                directions, layers = completion_dm_lastonly(
                    language_model, tokenizer, c["completions"], args.batch_size)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            # baseline_directions.pkl shape: {method: {...}}, so the evaluators'
            # load_directions(baseline_path=...) picks it up as-is
            with open(out_path, "wb") as f:
                pickle.dump({args.method: {"directions_per_layer": directions,
                                           "layers": layers, "sign": None}}, f)
            print(f"  wrote {out_path}")


def run_eval(concepts_by_category, args):
    for category, concepts in concepts_by_category.items():
        if category == "persona":
            import persona_steer_and_evaluate as evaluator
        elif category == "fear":
            import fear_steer_and_evaluate as evaluator
        else:
            import expert_steer_and_evaluate as evaluator
        summaries = []
        for c in concepts:
            out_dir = method_dir(args, category, c)
            directions_path = os.path.join(out_dir, "directions.pkl")
            summary = os.path.join(out_dir, "summary.csv")
            if not os.path.exists(directions_path):
                print(f"  SKIP {c['name']}: run --stage directions first")
                continue
            if not os.path.exists(summary):
                print(f"\n=== eval {args.method} | {category}: {c['name']} ===")
                evaluator.run_evaluation(
                    c["dataset"], directions_path, None, out_dir, args.model,
                    args.alpha_grid, judge_model=args.judge_model,
                    max_new_tokens=args.eval_gen_tokens, cache_dir=args.cache_dir, seed=args.seed,
                )
            else:
                print(f"  {summary} exists, skipping.")
            summaries.append(summary)

        combined = os.path.join(args.out_dir, f"combined_summary_{args.method}_{category}.csv")
        with open(combined, "w", newline="") as out_f:
            writer = csv.writer(out_f)
            header_written = False
            for summary in summaries:
                with open(summary) as in_f:
                    reader = csv.reader(in_f)
                    header = next(reader)
                    if not header_written:
                        writer.writerow(header)
                        header_written = True
                    writer.writerows(reader)
        print(f"Combined summary -> {combined}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["directions", "eval"])
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=CATEGORIES)
    p.add_argument("--model", required=True)
    p.add_argument("--model_tag", default="llama_8b",
                   help="matches <category>s_<model_tag>_alllayers in --self_out_dir")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--self_out_dir", required=True,
                   help="outputs root holding the self-completion runs (dataset_with_completions.json)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--alpha_grid", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    p.add_argument("--eval_gen_tokens", type=int, default=100)
    p.add_argument("--judge_model", default="gpt-4o-mini")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    concepts_by_category = {}
    for category in args.categories:
        concepts = concept_list(args.data_dir, args.self_out_dir, args.model_tag, category)
        missing = [c["completions"] for c in concepts if not os.path.exists(c["completions"])]
        if missing:
            raise FileNotFoundError(f"missing self-completion datasets, e.g. {missing[0]}")
        concepts_by_category[category] = concepts

    if args.stage == "directions":
        run_directions(concepts_by_category, args)
    else:
        run_eval(concepts_by_category, args)


if __name__ == "__main__":
    main()
