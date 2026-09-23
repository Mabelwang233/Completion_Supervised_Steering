"""
Runs the full pipeline independently for EACH fear concept in the
manifest -- matching Fig 4's per-concept design, where "fear of fire"
and "fear of ghosts" get entirely separate directions, not one shared
"fear" direction.

Mirrors persona_experiment.py exactly (same structure, same 7 methods,
same --completion_strategy / --completion_source support) -- only the
judge and dataset schema differ (fear_steer_and_evaluate.py / "fear"
field instead of "persona").

Everything that depends on completions (generate_completions, extract_
trajectories, sequence_rfm, bag_of_words_rfm, completion_rfm,
completion_diff_means, bag_of_words_diff_means, and the eval sweep) is
namespaced into a per-strategy (and, for teacher completions, per-source)
subdirectory. baselines (mean_difference, rfm) are the ONE exception:
they only ever read the prompt, never touch completions, so they're
strategy- and source-independent, computed ONCE per concept and shared
across everything else.

--completion_strategy {short_instructed_cut, long_subsampled_anchor}
--completion_source {self, teacher} -- "teacher" uses a stronger
external model (default o3-mini) via generate_completions_teacher.py,
applied to BOTH labels. Outputs go to a sibling '<strategy>_teacher/'
directory, never overwriting 'self' results.

Trains 7 methods per (concept, strategy, source): mean_difference, rfm
(shared baselines) + completion_rfm, completion_diff_means,
bag_of_words_rfm, bag_of_words_diff_means, sequence_rfm.
"""
import argparse
import json
import os

import generate_completions
import generate_completions_teacher
import extract_trajectories
import baselines
import train_sequence_rfm
import completion_rfm
import completion_diff_means
import bag_of_words_diff_means
import fear_steer_and_evaluate


def run_one_concept(fear, dataset_path, out_dir, args):
    os.makedirs(out_dir, exist_ok=True)
    strategy_tag = args.completion_strategy if args.completion_source == "self" else f"{args.completion_strategy}_teacher"
    strategy_dir = os.path.join(out_dir, strategy_tag)
    os.makedirs(strategy_dir, exist_ok=True)

    # shared across strategies AND sources -- never touches completions
    baseline_path = os.path.join(out_dir, "baseline_directions.pkl")

    # strategy-specific
    dataset_with_completions_path = os.path.join(strategy_dir, "dataset_with_completions.json")
    trajectories_path = os.path.join(strategy_dir, "trajectories.pt")
    seqrfm_path = os.path.join(strategy_dir, "sequence_rfm_directions.pt")
    bag_of_words_path = os.path.join(strategy_dir, "bag_of_words_rfm_directions.pt")
    completion_rfm_path = os.path.join(strategy_dir, "completion_rfm_directions.pkl")
    completion_dm_path = os.path.join(strategy_dir, "completion_diff_means_directions.pkl")
    bag_of_words_dm_path = os.path.join(strategy_dir, "bag_of_words_diff_means_directions.pt")

    print(f"\n{'='*60}\nFEAR: {fear}  |  STRATEGY: {args.completion_strategy}  |  SOURCE: {args.completion_source}\n{'='*60}")

    print("\n[1/9] train baselines (RFM, diff-means, all layers -- shared across strategies and sources)")
    if not os.path.exists(baseline_path):
        baselines.train_baselines(
            dataset_path, baseline_path, args.model,
            rfm_iters=args.rfm_iters, cache_dir=args.cache_dir, seed=args.seed,
        )
    else:
        print(f"  {baseline_path} exists, skipping.")

    print(f"\n[2/9] generate completions (source={args.completion_source}, "
          f"train, max {args.train_completion_tokens} tokens, "
          f"suffix={args.train_prompt_suffix!r})")
    if not os.path.exists(dataset_with_completions_path):
        if args.completion_source == "self":
            generate_completions.generate_completions(
                dataset_path, dataset_with_completions_path, args.model,
                cache_dir=args.cache_dir, max_new_tokens=args.train_completion_tokens, seed=args.seed,
                train_prompt_suffix=args.train_prompt_suffix,
            )
        else:  # "teacher"
            generate_completions_teacher.generate_completions_teacher(
                dataset_path, dataset_with_completions_path, target_model_name=args.model,
                teacher_model=args.teacher_model, max_new_tokens=args.train_completion_tokens,
                train_prompt_suffix=args.train_prompt_suffix,
                reasoning_effort=args.teacher_reasoning_effort,
                api_max_completion_tokens=args.teacher_api_max_completion_tokens,
                seed=args.seed,
            )
    else:
        print(f"  {dataset_with_completions_path} exists, skipping.")

    print(f"\n[3/9] extract activation trajectories "
          f"({'all layers' if args.layers is None else args.layers}, n_anchors={args.n_anchors})")
    if not os.path.exists(trajectories_path):
        extract_trajectories.extract_trajectories(
            dataset_with_completions_path, trajectories_path, args.model, args.layers,
            n_anchors=args.n_anchors, cache_dir=args.cache_dir, seed=args.seed,
        )
    else:
        print(f"  {trajectories_path} exists, skipping.")

    # print("\n[4/9] train sequence-RFM (per layer)")
    # if not os.path.exists(seqrfm_path):
    #     train_sequence_rfm.train(
    #         trajectories_path, seqrfm_path, sigma=args.sigma, lam=args.lam, k=args.k,
    #         rfm_iters=args.rfm_iters, ell_tau=args.ell_tau,
    #         temporal_weight=args.temporal_weight, kernel_type="gaussian",
    #         device="cuda", seed=args.seed,
    #     )
    # else:
    #     print(f"  {seqrfm_path} exists, skipping.")

    # print("\n[5/9] train bag-of-words RFM (k_tau===1, reuses trajectories.pt)")
    # if not os.path.exists(bag_of_words_path):
    #     train_sequence_rfm.train(
    #         trajectories_path, bag_of_words_path, sigma=args.sigma, lam=args.lam, k=args.k,
    #         rfm_iters=args.rfm_iters, ell_tau=args.ell_tau,
    #         temporal_weight=args.temporal_weight, kernel_type="gaussian",
    #         device="cuda", seed=args.seed, bag_of_words=True,
    #     )
    # else:
    #     print(f"  {bag_of_words_path} exists, skipping.")

    print("\n[6/9] train completion-RFM (last token of x,z)")
    if not os.path.exists(completion_rfm_path):
        completion_rfm.train_completion_rfm(
            dataset_with_completions_path, completion_rfm_path, args.model,
            rfm_iters=args.rfm_iters, cache_dir=args.cache_dir, seed=args.seed,
        )
    else:
        print(f"  {completion_rfm_path} exists, skipping.")

    # print("\n[7/9] train completion diff-means (last token of x,z, diff-means)")
    # if not os.path.exists(completion_dm_path):
    #     completion_diff_means.train_completion_diff_means(
    #         dataset_with_completions_path, completion_dm_path, args.model,
    #         cache_dir=args.cache_dir, seed=args.seed,
    #     )
    # else:
    #     print(f"  {completion_dm_path} exists, skipping.")

    print("\n[8/9] train bag-of-words diff-means (mean-pooled anchors, diff-means, reuses trajectories.pt)")
    if not os.path.exists(bag_of_words_dm_path):
        bag_of_words_diff_means.train_bag_of_words_diff_means(
            trajectories_path, bag_of_words_dm_path, seed=args.seed,
        )
    else:
        print(f"  {bag_of_words_dm_path} exists, skipping.")

    # Delete trajectories.pt now — it's only needed for steps 4/5/8 (all done).
    # Step 9 (eval) reads directions, not trajectories. Frees ~19 GB per concept.
    if os.path.exists(trajectories_path):
        os.remove(trajectories_path)
        print(f"  Deleted {trajectories_path} (~19 GB freed)")

    print(f"\n[9/9] steer + evaluate (alpha sweep, fear judge, "
          f"max {args.eval_gen_tokens} tokens/response, 7 methods)")
    fear_steer_and_evaluate.run_evaluation(
        dataset_path, baseline_path, None, strategy_dir, args.model,
        args.alpha_grid, judge_model=args.judge_model, max_new_tokens=args.eval_gen_tokens,
        cache_dir=args.cache_dir, seed=args.seed,
        completion_rfm_path=completion_rfm_path,
        bag_of_words_path=None,
        completion_diff_means_path=None,
        bag_of_words_diff_means_path=bag_of_words_dm_path,
    )


def main(args):
    with open(args.manifest_path) as f:
        manifest = json.load(f)

    strategy_tag = args.completion_strategy if args.completion_source == "self" else f"{args.completion_strategy}_teacher"

    all_summaries = []
    for entry in manifest:
        fear = entry["fear"]
        dataset_path = entry["path"]
        safe_name = fear.lower().replace(" ", "_")
        out_dir = os.path.join(args.output_dir, f"fear_{safe_name}")
        run_one_concept(fear, dataset_path, out_dir, args)
        all_summaries.append(os.path.join(out_dir, strategy_tag, "summary.csv"))

    # combine all per-concept summaries into one file (per strategy x source)
    import csv
    combined_path = os.path.join(args.output_dir, f"combined_summary_{strategy_tag}.csv")
    with open(combined_path, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["fear", "method", "alpha", "mean_score", "n"])
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
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=1e-3)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--n_anchors", type=int, default=25)
    p.add_argument("--ell_tau", type=float, default=0.15)
    p.add_argument("--temporal_weight", default="uniform", choices=["uniform", "linear_late"])
    p.add_argument("--alpha_grid", type=float, nargs="+", required=True)
    p.add_argument("--completion_strategy", required=True,
                    choices=["short_instructed_cut", "long_subsampled_anchor"],
                    help="short_instructed_cut: length instruction + hard cap on training completions. "
                         "long_subsampled_anchor: full-length completions, anchor-subsampled after the fact.")
    p.add_argument("--train_prompt_suffix", default=None,
                    help='e.g. "Answer in 15 words or less." for short_instructed_cut; '
                         'omit (or pass nothing) for long_subsampled_anchor')
    p.add_argument("--completion_source", default="self", choices=["self", "teacher"],
                    help="self (default): target model generates its own training completions. "
                         "teacher: a stronger external model (--teacher_model) generates them instead, "
                         "via the OpenAI API -- applied to both labels. Outputs go to a sibling "
                         "'<strategy>_teacher/' directory, never overwriting 'self' results.")
    p.add_argument("--teacher_model", default="o3-mini",
                    help="only used when --completion_source teacher")
    p.add_argument("--teacher_reasoning_effort", default="low", choices=["low", "medium", "high"],
                    help="only used when --completion_source teacher")
    p.add_argument("--teacher_api_max_completion_tokens", type=int, default=1000,
                    help="only used when --completion_source teacher -- the ACTUAL API request budget, "
                         "must cover hidden reasoning tokens plus visible output. Deliberately decoupled "
                         "from --train_completion_tokens. Control visible answer length via "
                         "--train_prompt_suffix instead.")
    p.add_argument("--train_completion_tokens", type=int, default=15,
                    help="max new tokens when generating TRAINING completions (z). "
                         "15 for short_instructed_cut, ~100 for long_subsampled_anchor.")
    p.add_argument("--eval_gen_tokens", type=int, default=100,
                    help="max new tokens when generating TEST-TIME responses to be judged.")
    p.add_argument("--judge_model", default="gpt-4o-mini")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    main(args)