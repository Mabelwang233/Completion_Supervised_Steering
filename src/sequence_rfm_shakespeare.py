"""
sequence_rfm_shakespeare.py -- main entry point, run via run.sh.

Matches the neural_controllers notebook convention: directions trained
and applied at ALL layers (not a single layer), same coefficient at
every layer simultaneously during steering.

Orchestrates:
  1. data_prep            (build (x, y) pairs -- skipped if dataset exists)
  2. generate_completions (z_i via the target model itself)
  3. extract_trajectories (all layers, anchor-subsampled)
  4. train_baselines      (RFM + diff-means, prompt-only, all layers, existing toolkits)
  5. train_sequence_rfm   (kernel-mean K_M + cross-time AGOP, PER LAYER -- the expensive step)
  6. steer_and_evaluate   (alpha sweep x 3 methods x 80 test prompts, all layers, judged)

--layers controls which layers get trained/steered. Omit for all layers
(notebook default, ~30 for an 8B model, ~15-30h for step 5 -- see that
file's docstring). Pass a subset (e.g. every 3rd layer) to cut cost
without changing any code.
"""
import argparse
import os

import data_prep
import generate_completions
import extract_trajectories
import baselines
import train_sequence_rfm
import completion_rfm
import steer_and_evaluate


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    dataset_path = os.path.join(args.data_path, "shakespeare_dataset.json")
    dataset_with_completions_path = os.path.join(args.output_dir, "shakespeare_dataset_with_completions.json")
    trajectories_path = os.path.join(args.output_dir, "trajectories.pt")
    baseline_path = os.path.join(args.output_dir, "baseline_directions.pkl")
    seqrfm_path = os.path.join(args.output_dir, "sequence_rfm_directions.pt")
    completion_rfm_path = os.path.join(args.output_dir, "completion_rfm_directions.pkl")
    bag_of_words_path = os.path.join(args.output_dir, "bag_of_words_rfm_directions.pt")

    print("\n[1/8] data prep")
    if not os.path.exists(dataset_path):
        data_prep.main(data_dir=args.data_path, out_path=dataset_path)
    else:
        print(f"  {dataset_path} already exists, skipping.")

    print("\n[2/8] generate completions (target model, unsteered, greedy)")
    if not os.path.exists(dataset_with_completions_path):
        generate_completions.generate_completions(
            dataset_path, dataset_with_completions_path, args.model,
            cache_dir=args.cache_dir, max_new_tokens=args.train_completion_tokens, seed=args.seed,
        )
    else:
        print(f"  {dataset_with_completions_path} already exists, skipping.")

    print(f"\n[3/8] extract activation trajectories ({'all layers' if args.layers is None else args.layers})")
    if not os.path.exists(trajectories_path):
        extract_trajectories.extract_trajectories(
            dataset_with_completions_path, trajectories_path, args.model, args.layers,
            n_anchors=args.n_anchors, cache_dir=args.cache_dir, seed=args.seed,
        )
    else:
        print(f"  {trajectories_path} already exists, skipping.")

    print("\n[4/8] train baselines (RFM, diff-means, all layers)")
    if not os.path.exists(baseline_path):
        baselines.train_baselines(
            dataset_path, baseline_path, args.model,
            rfm_iters=args.rfm_iters, cache_dir=args.cache_dir, seed=args.seed,
        )
    else:
        print(f"  {baseline_path} already exists, skipping.")

    print("\n[5/8] train sequence-RFM (per layer -- this is the expensive step)")
    if not os.path.exists(seqrfm_path):
        train_sequence_rfm.train(
            trajectories_path, seqrfm_path, sigma=args.sigma, lam=args.lam, k=args.k,
            rfm_iters=args.rfm_iters, ell_tau=args.ell_tau,
            temporal_weight=args.temporal_weight, kernel_type="gaussian",
            device="cuda", seed=args.seed,
        )
    else:
        print(f"  {seqrfm_path} already exists, skipping.")

    print("\n[6/8] train completion-RFM (last token of x,z -- between vanilla and sequence RFM)")
    if not os.path.exists(completion_rfm_path):
        completion_rfm.train_completion_rfm(
            dataset_with_completions_path, completion_rfm_path, args.model,
            rfm_iters=args.rfm_iters, cache_dir=args.cache_dir, seed=args.seed,
        )
    else:
        print(f"  {completion_rfm_path} already exists, skipping.")

    print("\n[7/8] train bag-of-words RFM (k_tau===1, position ignored -- reuses trajectories.pt)")
    if not os.path.exists(bag_of_words_path):
        train_sequence_rfm.train(
            trajectories_path, bag_of_words_path, sigma=args.sigma, lam=args.lam, k=args.k,
            rfm_iters=args.rfm_iters, ell_tau=args.ell_tau,
            temporal_weight=args.temporal_weight, kernel_type="gaussian",
            device="cuda", seed=args.seed, bag_of_words=True,
        )
    else:
        print(f"  {bag_of_words_path} already exists, skipping.")

    print("\n[8/8] steer + evaluate (alpha sweep, all layers, GPT judge, 5 methods)")
    steer_and_evaluate.run_evaluation(
        dataset_path, baseline_path, seqrfm_path, args.output_dir, args.model,
        args.alpha_grid, judge_model=args.judge_model, max_new_tokens=args.eval_gen_tokens,
        cache_dir=args.cache_dir, seed=args.seed,
        completion_rfm_path=completion_rfm_path,
        bag_of_words_path=bag_of_words_path,
    )

    print(f"\nDone. Results in {args.output_dir}/summary.csv and per_example_results.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data_path", required=True, help="directory containing class_0.txt / class_1.txt")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--layers", nargs="+", default=None,
                    help="explicit layer indices, or omit for all layers (notebook default)")
    p.add_argument("--rfm_iters", type=int, default=3)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=1e-3)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--n_anchors", type=int, default=25)
    p.add_argument("--ell_tau", type=float, default=0.15)
    p.add_argument("--temporal_weight", default="uniform", choices=["uniform", "linear_late"])
    p.add_argument("--alpha_grid", type=float, nargs="+", required=True)
    p.add_argument("--train_completion_tokens", type=int, default=15,
                    help="max new tokens when generating TRAINING completions (z).")
    p.add_argument("--eval_gen_tokens", type=int, default=100,
                    help="max new tokens for TEST-TIME responses to be judged.")
    p.add_argument("--judge_model", default="gpt-4o-mini")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    main(args)
