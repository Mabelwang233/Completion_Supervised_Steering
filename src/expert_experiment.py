"""
Runs the pipeline independently for EACH expert concept in the manifest,
mirroring persona_experiment.py (same active steps, same per-strategy
directory layout), judged with expert_judge.

Active methods: mean_difference, rfm (shared baselines), completion_rfm,
bag_of_words_diff_means. sequence_rfm / bag_of_words_rfm are run by
expert_experiment_seqrfm.py, as for personas.

Output layout:
  <output_dir>/expert_<name>/baseline_directions.pkl
  <output_dir>/expert_<name>/<completion_strategy>/{dataset_with_completions.json, ..., summary.csv}
  <output_dir>/combined_summary_<completion_strategy>.csv
"""
import argparse
import csv
import json
import os

import generate_completions
import extract_trajectories
import baselines
import completion_rfm
import bag_of_words_diff_means
import expert_steer_and_evaluate
from timing import timed


def safe_name(expert):
    return expert.lower().replace(" ", "_")


def run_one_concept(expert, dataset_path, out_dir, args):
    os.makedirs(out_dir, exist_ok=True)
    strategy_dir = os.path.join(out_dir, args.completion_strategy)
    os.makedirs(strategy_dir, exist_ok=True)

    # shared across strategies -- never touches completions
    baseline_path = os.path.join(out_dir, "baseline_directions.pkl")

    dataset_with_completions_path = os.path.join(strategy_dir, "dataset_with_completions.json")
    trajectories_path = os.path.join(strategy_dir, "trajectories.pt")
    completion_rfm_path = os.path.join(strategy_dir, "completion_rfm_directions.pkl")
    bag_of_words_dm_path = os.path.join(strategy_dir, "bag_of_words_diff_means_directions.pt")
    # training-time wall clock per stage (includes model load; evaluation not timed)
    timings_path = os.path.join(out_dir, "timings.json")

    print(f"\n{'='*60}\nEXPERT: {expert}  |  STRATEGY: {args.completion_strategy}\n{'='*60}")
    if os.path.exists(os.path.join(strategy_dir, "summary.csv")):
        print(f"  {os.path.join(strategy_dir, 'summary.csv')} exists, all steps done, skipping.")
        return

    print("\n[1/6] train baselines (RFM, diff-means, all layers -- shared across strategies)")
    if not os.path.exists(baseline_path):
        baselines.train_baselines(
            dataset_path, baseline_path, args.model,
            rfm_iters=args.rfm_iters, cache_dir=args.cache_dir, seed=args.seed,
            timings_path=timings_path,
        )
    else:
        print(f"  {baseline_path} exists, skipping.")

    print(f"\n[2/6] generate completions (train, max {args.train_completion_tokens} tokens, "
          f"suffix={args.train_prompt_suffix!r})")
    if not os.path.exists(dataset_with_completions_path):
        with timed(timings_path, "generate_completions"):
            generate_completions.generate_completions(
                dataset_path, dataset_with_completions_path, args.model,
                cache_dir=args.cache_dir, max_new_tokens=args.train_completion_tokens, seed=args.seed,
                train_prompt_suffix=args.train_prompt_suffix,
            )
    else:
        print(f"  {dataset_with_completions_path} exists, skipping.")

    print(f"\n[3/6] train completion-RFM (last token of x,z)")
    if not os.path.exists(completion_rfm_path):
        with timed(timings_path, "completion_rfm"):
            completion_rfm.train_completion_rfm(
                dataset_with_completions_path, completion_rfm_path, args.model,
                rfm_iters=args.rfm_iters, cache_dir=args.cache_dir, seed=args.seed,
            )
    else:
        print(f"  {completion_rfm_path} exists, skipping.")

    print(f"\n[4/6] extract activation trajectories "
          f"({'all layers' if args.layers is None else args.layers}, n_anchors={args.n_anchors})")
    if not os.path.exists(bag_of_words_dm_path) and not os.path.exists(trajectories_path):
        with timed(timings_path, "extract_trajectories"):
            extract_trajectories.extract_trajectories(
                dataset_with_completions_path, trajectories_path, args.model, args.layers,
                n_anchors=args.n_anchors, cache_dir=args.cache_dir, seed=args.seed,
            )
    else:
        print("  trajectories.pt exists or not needed, skipping.")

    print("\n[5/6] train bag-of-words diff-means (mean-pooled anchors, reuses trajectories.pt)")
    if not os.path.exists(bag_of_words_dm_path):
        with timed(timings_path, "bag_of_words_diff_means"):
            bag_of_words_diff_means.train_bag_of_words_diff_means(
                trajectories_path, bag_of_words_dm_path, seed=args.seed,
            )
    else:
        print(f"  {bag_of_words_dm_path} exists, skipping.")

    # Only bag-of-words diff-means reads trajectories here; delete to free ~19 GB.
    # expert_experiment_seqrfm.py re-extracts them if needed.
    if os.path.exists(trajectories_path) and not args.keep_trajectories:
        os.remove(trajectories_path)
        print(f"  Deleted {trajectories_path} (~19 GB freed)")

    print(f"\n[6/6] steer + evaluate (alpha sweep, expert judge, "
          f"max {args.eval_gen_tokens} tokens/response)")
    # summary.csv is written only after the whole sweep is judged, so it marks a finished eval
    if os.path.exists(os.path.join(strategy_dir, "summary.csv")):
        print("  summary.csv exists, skipping.")
        return
    expert_steer_and_evaluate.run_evaluation(
        dataset_path, baseline_path, None, strategy_dir, args.model,
        args.alpha_grid, judge_model=args.judge_model, max_new_tokens=args.eval_gen_tokens,
        cache_dir=args.cache_dir, seed=args.seed,
        completion_rfm_path=completion_rfm_path,
        bag_of_words_diff_means_path=bag_of_words_dm_path,
    )


def main(args):
    with open(args.manifest_path) as f:
        manifest = json.load(f)

    all_summaries = []
    for entry in manifest:
        expert = entry["expert"]
        out_dir = os.path.join(args.output_dir, f"expert_{safe_name(expert)}")
        run_one_concept(expert, entry["path"], out_dir, args)
        all_summaries.append(os.path.join(out_dir, args.completion_strategy, "summary.csv"))

    combined_path = os.path.join(args.output_dir, f"combined_summary_{args.completion_strategy}.csv")
    with open(combined_path, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["expert", "method", "alpha", "mean_score", "n"])
        for summary_path in all_summaries:
            if not os.path.exists(summary_path):
                continue
            with open(summary_path) as in_f:
                reader = csv.reader(in_f)
                next(reader)  # skip header
                for row in reader:
                    writer.writerow(row)

    print(f"\nAll concepts done. Combined summary: {combined_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest_path", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--layers", nargs="+", default=None,
                    help="explicit layer indices, or omit for all layers")
    p.add_argument("--rfm_iters", type=int, default=3)
    p.add_argument("--n_anchors", type=int, default=100)
    p.add_argument("--keep_trajectories", action="store_true",
                    help="keep trajectories.pt so expert_experiment_seqrfm.py can reuse it")
    p.add_argument("--alpha_grid", type=float, nargs="+", required=True)
    p.add_argument("--completion_strategy", required=True,
                    choices=["short_instructed_cut", "long_subsampled_anchor"])
    p.add_argument("--train_prompt_suffix", default=None,
                    help='e.g. "Answer in 15 words or less." for short_instructed_cut; '
                         'omit for long_subsampled_anchor')
    p.add_argument("--train_completion_tokens", type=int, default=100)
    p.add_argument("--eval_gen_tokens", type=int, default=100)
    p.add_argument("--judge_model", default="gpt-4o-mini")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    main(args)
