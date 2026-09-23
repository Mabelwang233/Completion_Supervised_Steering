"""
Train + evaluate CLAS for one concept (persona or fear). Add-on script,
same pattern as run_linear_late.sh: reuses already-computed artifacts
(baseline_directions.pkl for d_l, an existing dataset_with_completions.json
for the steer pairs -- intended to be the teacher/long_subsampled_anchor
one) rather than regenerating anything.

No alpha sweep -- CLAS's whole point is a LEARNED, context-dependent
coefficient, not a hand-tuned global one. Produces a single score per
concept (alpha column set to 0.0 as a documented placeholder, kept for
schema compatibility with the other methods' summary.csv files).
"""
import argparse
import csv
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, ".")
from utils import load_model
import clas


def run_one_concept(concept_type, concept_name, dataset_path, dataset_with_completions_path,
                     baseline_path, out_dir, model_name, judge_model="gpt-4o-mini",
                     max_new_tokens=100, val_frac=0.2, lr_weight=3e-3, lr_bias=1e-1,
                     epochs=8, accum_steps=10, patience=2, cache_dir=None, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    clas_path = os.path.join(out_dir, "clas_directions.pt")

    print(f"\n{'='*60}\n{concept_type.upper()}: {concept_name}  |  METHOD: clas\n{'='*60}")

    print("\n[1/2] train CLAS sensing vectors (freeze LLM, learn c_l per layer)")
    if not os.path.exists(clas_path):
        clas.train_clas(
            dataset_with_completions_path, baseline_path, clas_path, model_name,
            val_frac=val_frac, lr_weight=lr_weight, lr_bias=lr_bias,
            epochs=epochs, accum_steps=accum_steps, patience=patience,
            cache_dir=cache_dir, seed=seed,
        )
    else:
        print(f"  {clas_path} exists, skipping.")

    print("\n[2/2] generate + evaluate on the 50 fixed test questions (no alpha sweep)")
    saved = torch.load(clas_path)
    d_directions = saved["d_directions_per_layer"]
    c_weight = saved["c_weight_per_layer"]
    c_bias = saved["c_bias_per_layer"]

    with open(dataset_path) as f:
        dataset = json.load(f)
    test_examples = dataset["test"]

    if concept_type == "persona":
        from persona_judge import judge_response
        concept_field = "persona"
    elif concept_type == "topophile":
        from topophile_judge import judge_response
        concept_field = "concept"
    else:
        from fear_judge import judge_response
        concept_field = "fear"
    target = dataset[concept_field]

    language_model, tokenizer = load_model(model_name, cache_dir=cache_dir)
    language_model.eval()

    rows_path = os.path.join(out_dir, "per_example_results.csv")
    summary_path = os.path.join(out_dir, "summary.csv")

    scores = []
    with open(rows_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([concept_field, "method", "alpha", "question", "response", "score", "explanation"])
        for ex in tqdm(test_examples):
            question = ex.get("question", ex.get("prompt"))
            response = clas.generate_with_clas(
                language_model, tokenizer, ex["prompt"], d_directions, c_weight, c_bias, max_new_tokens
            )
            score, explanation = judge_response(target, question, response, judge_model)
            writer.writerow([target, "clas", 0.0, question, response, score, explanation])
            scores.append(score)

    mean_score = sum(scores) / len(scores)
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([concept_field, "method", "alpha", "mean_score", "n"])
        # alpha=0.0 is a placeholder -- CLAS has no alpha sweep, see module docstring
        writer.writerow([target, "clas", 0.0, mean_score, len(scores)])

    print(f"\nCLAS mean_score for {target}: {mean_score:.3f}  (n={len(scores)})")
    print(f"Wrote {rows_path}")
    print(f"Wrote {summary_path}")


def main(args):
    with open(args.manifest_path) as f:
        manifest = json.load(f)
    if args.concept_type == "persona":
        field = "persona"
    elif args.concept_type == "topophile":
        field = "concept"
    else:
        field = "fear"

    all_summaries = []
    for entry in manifest:
        concept_name = entry[field]
        dataset_path = entry["path"]
        safe_name = concept_name.lower().replace(" ", "_").replace(".", "").replace(",", "")
        concept_out_dir = os.path.join(args.output_dir, f"{args.concept_type}_{safe_name}")
        dataset_with_completions_path = os.path.join(
            concept_out_dir, args.source_strategy_dir, "dataset_with_completions.json"
        )
        baseline_path = os.path.join(concept_out_dir, "baseline_directions.pkl")
        clas_out_dir = os.path.join(concept_out_dir, f"clas_{args.source_strategy_dir}")

        if not os.path.exists(dataset_with_completions_path):
            print(f"SKIPPING {concept_name}: {dataset_with_completions_path} not found -- "
                  f"run the teacher completion-strategy pipeline for this concept first.")
            continue
        if not os.path.exists(baseline_path):
            print(f"SKIPPING {concept_name}: {baseline_path} not found -- "
                  f"run the main pipeline for this concept first (baselines are shared, computed there).")
            continue

        run_one_concept(
            args.concept_type, concept_name, dataset_path, dataset_with_completions_path,
            baseline_path, clas_out_dir, args.model, judge_model=args.judge_model,
            max_new_tokens=args.eval_gen_tokens, val_frac=args.val_frac,
            lr_weight=args.lr_weight, lr_bias=args.lr_bias, epochs=args.epochs,
            accum_steps=args.accum_steps, patience=args.patience,
            cache_dir=args.cache_dir, seed=args.seed,
        )
        all_summaries.append(os.path.join(clas_out_dir, "summary.csv"))

    import csv as csv_mod
    combined_path = os.path.join(args.output_dir, f"combined_summary_clas_{args.source_strategy_dir}.csv")
    with open(combined_path, "w", newline="") as out_f:
        writer = csv_mod.writer(out_f)
        writer.writerow([field, "method", "alpha", "mean_score", "n"])
        for summary_path in all_summaries:
            if not os.path.exists(summary_path):
                continue
            with open(summary_path) as in_f:
                reader = csv_mod.reader(in_f)
                next(reader)
                for row in reader:
                    writer.writerow(row)
    print(f"\nAll concepts done. Combined CLAS summary: {combined_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--concept_type", required=True, choices=["persona", "fear", "topophile"])
    p.add_argument("--manifest_path", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--source_strategy_dir", default="long_subsampled_anchor_teacher",
                    help="which existing strategy subfolder to pull the steer-dataset completions from")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--val_frac", type=float, default=0.2,
                    help="fraction of the cross-paired steer dataset held out for val-loss-based "
                         "best-checkpoint selection and early stopping -- matches "
                         "train_controller.py's completion-RFM+controller convention")
    p.add_argument("--lr_weight", type=float, default=3e-3)
    p.add_argument("--lr_bias", type=float, default=1e-1)
    p.add_argument("--epochs", type=int, default=8,
                    help="read as EPOCHS, not the paper's literal step count -- see "
                         "clas.py's train_clas docstring for why")
    p.add_argument("--accum_steps", type=int, default=10,
                    help="gradient-accumulation steps at batch_size=1 -- effective batch size, "
                         "CLAS-aligned default of 10, fixed regardless of dataset size")
    p.add_argument("--patience", type=int, default=2,
                    help="stop after this many consecutive epochs with no val_loss improvement")
    p.add_argument("--eval_gen_tokens", type=int, default=100)
    p.add_argument("--judge_model", default="gpt-4o-mini")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    main(args)