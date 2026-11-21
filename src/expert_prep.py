"""
Steps 1-2 of expert_experiment.py only: prompt baselines (baseline_directions.pkl,
needed by CLAS) and self-completions (dataset_with_completions.json, needed by
contextual CLAS and joint training). Same calls and arguments as
expert_experiment.run_one_concept, without the alpha-sweep methods/eval.

Usage:
    python expert_prep.py --manifest_path M --model X --output_dir OUT [--cache_dir C]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines
import generate_completions
from timing import timed
from expert_experiment import safe_name


def main(args):
    with open(args.manifest_path) as f:
        manifest = json.load(f)
    for entry in manifest:
        expert, dataset_path = entry["expert"], entry["path"]
        out_dir = os.path.join(args.output_dir, f"expert_{safe_name(expert)}")
        strategy_dir = os.path.join(out_dir, args.completion_strategy)
        os.makedirs(strategy_dir, exist_ok=True)
        baseline_path = os.path.join(out_dir, "baseline_directions.pkl")
        completions_path = os.path.join(strategy_dir, "dataset_with_completions.json")
        timings_path = os.path.join(out_dir, "timings.json")
        print(f"\n{'='*60}\nEXPERT: {expert}\n{'='*60}")

        print("[1/2] train baselines (RFM, diff-means, all layers)")
        if not os.path.exists(baseline_path):
            baselines.train_baselines(
                dataset_path, baseline_path, args.model,
                rfm_iters=args.rfm_iters, cache_dir=args.cache_dir, seed=args.seed,
                timings_path=timings_path,
            )
        else:
            print(f"  {baseline_path} exists, skipping.")

        print(f"[2/2] generate completions (max {args.train_completion_tokens} tokens)")
        if not os.path.exists(completions_path):
            with timed(timings_path, "generate_completions"):
                generate_completions.generate_completions(
                    dataset_path, completions_path, args.model,
                    cache_dir=args.cache_dir, max_new_tokens=args.train_completion_tokens,
                    seed=args.seed, train_prompt_suffix=None,
                )
        else:
            print(f"  {completions_path} exists, skipping.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest_path", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--rfm_iters", type=int, default=3)
    p.add_argument("--completion_strategy", default="long_subsampled_anchor")
    p.add_argument("--train_completion_tokens", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
