"""
Expert-concept version of persona_steer_and_evaluate.py -- same steering
logic (all layers, unit-norm directions, single global alpha), judged
with expert_judge.judge_response(expert, question, response).

load_directions / generate_steered are imported from the persona
version rather than copied, so steering stays identical across classes.

Test set is the fixed list of 20 expert-eliciting questions.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import torch
from tqdm import tqdm

sys.path.insert(0, ".")
from utils import load_model
from expert_judge import judge_response
from persona_steer_and_evaluate import load_directions, generate_steered


def run_evaluation(dataset_path, baseline_path, seqrfm_path, out_dir, model_name,
                    alpha_grid, judge_model="gpt-4o-mini", max_new_tokens=100,
                    cache_dir=None, seed=0, completion_rfm_path=None, bag_of_words_path=None,
                    completion_diff_means_path=None, bag_of_words_diff_means_path=None,
                    seqrfm_method_name="sequence_rfm"):
    torch.manual_seed(seed)

    with open(dataset_path) as f:
        dataset = json.load(f)
    expert = dataset["expert"]
    test_examples = dataset["test"]  # 20 fixed questions, no held-out statements

    language_model, tokenizer = load_model(model_name, cache_dir=cache_dir)
    language_model.eval()

    directions = load_directions(baseline_path, seqrfm_path, completion_rfm_path, bag_of_words_path,
                                  completion_diff_means_path, bag_of_words_diff_means_path,
                                  seqrfm_method_name=seqrfm_method_name)
    print(f"Expert: {expert}")
    for method, per_layer in directions.items():
        print(f"  {method}: directions at {len(per_layer)} layers")
    print(f"  alpha grid: {alpha_grid}")

    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, "per_example_results.csv")
    summary_path = os.path.join(out_dir, "summary.csv")

    all_rows = []
    with open(rows_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["expert", "method", "alpha", "question", "response", "score", "explanation"])

        for method, per_layer_directions in directions.items():
            for alpha in alpha_grid:
                print(f"\n=== expert={expert}  method={method}  alpha={alpha} ===")
                for ex in tqdm(test_examples):
                    response = generate_steered(language_model, tokenizer, ex["prompt"],
                                                 per_layer_directions, alpha, max_new_tokens)
                    score, explanation = judge_response(expert, ex["question"], response, judge_model)
                    writer.writerow([expert, method, alpha, ex["question"], response, score, explanation])
                    all_rows.append({"method": method, "alpha": alpha, "score": score})

    agg = defaultdict(list)
    for r in all_rows:
        agg[(r["method"], r["alpha"])].append(r["score"])

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["expert", "method", "alpha", "mean_score", "n"])
        for (method, alpha), scores in sorted(agg.items()):
            writer.writerow([expert, method, alpha, sum(scores) / len(scores), len(scores)])

    print(f"\nWrote {rows_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--baseline_path", default=None,
                    help="optional: pickle from baselines.py, adds mean_difference + rfm to the sweep")
    p.add_argument("--seqrfm_path", default=None,
                    help="optional: .pt from train_sequence_rfm.py, adds a sequence-RFM variant to the sweep")
    p.add_argument("--seqrfm_method_name", default="sequence_rfm")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--alpha_grid", type=float, nargs="+", required=True)
    p.add_argument("--judge_model", default="gpt-4o-mini")
    p.add_argument("--eval_n_tokens", type=int, default=100)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--completion_rfm_path", default=None)
    p.add_argument("--bag_of_words_path", default=None)
    p.add_argument("--completion_diff_means_path", default=None)
    p.add_argument("--bag_of_words_diff_means_path", default=None)
    args = p.parse_args()

    run_evaluation(args.dataset_path, args.baseline_path, args.seqrfm_path, args.out_dir,
                    args.model, args.alpha_grid, args.judge_model,
                    args.eval_n_tokens, args.cache_dir, args.seed,
                    completion_rfm_path=args.completion_rfm_path,
                    bag_of_words_path=args.bag_of_words_path,
                    completion_diff_means_path=args.completion_diff_means_path,
                    bag_of_words_diff_means_path=args.bag_of_words_diff_means_path,
                    seqrfm_method_name=args.seqrfm_method_name)
