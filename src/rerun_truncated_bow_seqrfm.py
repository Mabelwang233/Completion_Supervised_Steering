"""
Refits bag-of-words-RFM and sequence-RFM with truncate_at_end=True for
every concept in a manifest, reusing EXISTING completions/trajectories
(no regeneration), and evaluates ONLY those 2 methods into a fresh
summary CSV -- so it can be merged with your existing 7-method combined
CSV without touching the other 5 methods' scores.

Works for both fear and persona concepts via --concept_type. Mirrors
fear_experiment.py's directory/manifest conventions; the persona side
is carried over from run_fear_controller_pipeline.py's docstring
("persona manifest uses {'persona': ..., 'path': ...} entries,
persona_<name>/ directories") since persona_experiment.py itself
wasn't available to check directly -- verify this before trusting a
full 20-concept run (see the printed [check] line for concept #1).

COST WARNING (read before running for all 20 concepts):
  --bag_of_words + --fast_bow: cheap, GPU-batched. truncate_at_end
    barely changes the runtime (matches what you already measured).
  Sequence-RFM + --fast_bow: now ALSO batched (kernels_batched_positional.py),
    generalizing the bag-of-words fast path to the full k_tau-weighted
    kernel. Verified against the loop reference to floating-point
    precision on synthetic data (see verify_fast_bow_positional.py) --
    but you should run that verification script yourself before trusting
    real results, and time ONE concept before committing to all 20:
    truncate_at_end still multiplies intermediate-iteration cost by
    roughly (n_train/k)x regardless of fast_bow, so the two flags'
    costs don't simply cancel out. --only_bow remains available to
    skip sequence-RFM entirely if that turns out too expensive.

Requires (per concept, under --old_output_dir, from your EXISTING
fear_experiment.py / persona_experiment.py run):
    <old_output_dir>/{fear,persona}_<name>/<strategy>/trajectories.pt
    <old_output_dir>/{fear,persona}_<name>/<strategy>/dataset_with_completions.json

Writes (per concept, under --new_output_dir -- never touches old_output_dir):
    <new_output_dir>/{fear,persona}_<name>/<strategy>/sequence_rfm_directions.pt
    <new_output_dir>/{fear,persona}_<name>/<strategy>/bag_of_words_rfm_directions.pt
    <new_output_dir>/{fear,persona}_<name>/<strategy>/summary.csv   (2 methods only)
    <new_output_dir>/combined_summary_{concept_type}_{strategy}_truncated.csv

Usage:
    python rerun_truncated_bow_seqrfm.py \\
        --concept_type fear \\
        --manifest_path $DATA_DIR/fears/manifest.json \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --old_output_dir $OUT_DIR/fears_llama_8b_alllayers \\
        --new_output_dir $OUT_DIR/fears_llama_8b_alllayers_truncated \\
        --alpha_grid 0.3 0.4 0.55 0.6 0.65 0.7 0.75 0.8 0.9 \\
        --fast_bow \\
        --cache_dir $CACHE_DIR
"""
import argparse
import csv
import json
import os

import train_sequence_rfm

CONCEPT_CONFIG = {
    "fear": {
        "manifest_key": "fear",
        "dir_prefix": "fear",
        # verified from run_fear_controller_pipeline.py / fear_experiment.py
    },
    "persona": {
        "manifest_key": "persona",
        "dir_prefix": "persona",
        # NOT independently verified -- carried over by analogy from
        # run_fear_controller_pipeline.py's docstring. Check the
        # printed [check] line for the first concept before trusting
        # this for all 20.
    },
}


def safe_name(name):
    return name.lower().replace(" ", "_")


def run_one(concept_name, dataset_path, old_strategy_dir, new_strategy_dir, args):
    trajectories_path = os.path.join(old_strategy_dir, "trajectories.pt")
    completions_path = os.path.join(old_strategy_dir, "dataset_with_completions.json")

    if not os.path.exists(trajectories_path) or not os.path.exists(completions_path):
        missing = "trajectories.pt" if not os.path.exists(trajectories_path) else "dataset_with_completions.json"
        print(f"  SKIP {concept_name}: missing {missing} under {old_strategy_dir}")
        return None

    os.makedirs(new_strategy_dir, exist_ok=True)
    seqrfm_path = os.path.join(new_strategy_dir, "sequence_rfm_directions.pt")
    bag_of_words_path = os.path.join(new_strategy_dir, "bag_of_words_rfm_directions.pt")

    print(f"\n--- {concept_name} ---")

    if not args.only_seqrfm:
        print(f"[1/{'2' if not args.only_bow else '1'}] bag-of-words RFM (truncate_at_end=True"
              f"{', fast_bow' if args.fast_bow else ''})")
        if not os.path.exists(bag_of_words_path):
            train_sequence_rfm.train(
                trajectories_path, bag_of_words_path, sigma=args.sigma, lam=args.lam, k=args.k,
                rfm_iters=args.rfm_iters, ell_tau=args.ell_tau,
                temporal_weight=args.temporal_weight, kernel_type="gaussian",
                device=args.device, seed=args.seed, bag_of_words=True,
                truncate_at_end=True, fast_bow=args.fast_bow,
                fast_bow_chunk_size=args.fast_bow_chunk_size,
            )
        else:
            print(f"  {bag_of_words_path} exists, skipping.")

    if not args.only_bow:
        print(f"[{'2' if not args.only_seqrfm else '1'}/{'2' if not args.only_seqrfm else '1'}] "
              f"sequence-RFM (truncate_at_end=True{', fast_bow' if args.fast_bow else ''})")
        if not os.path.exists(seqrfm_path):
            train_sequence_rfm.train(
                trajectories_path, seqrfm_path, sigma=args.sigma, lam=args.lam, k=args.k,
                rfm_iters=args.rfm_iters, ell_tau=args.ell_tau,
                temporal_weight=args.temporal_weight, kernel_type="gaussian",
                device=args.device, seed=args.seed, bag_of_words=False,
                truncate_at_end=True, fast_bow=args.fast_bow,
                fast_bow_chunk_size=args.fast_bow_chunk_size,
            )
        else:
            print(f"  {seqrfm_path} exists, skipping.")

    print("[eval] steer + evaluate (alpha sweep, 2 methods only)")
    steer_and_evaluate = args.steer_and_evaluate_module
    steer_and_evaluate.run_evaluation(
        dataset_path,
        baseline_path=None,  # explicitly omit -- do NOT re-evaluate mean_difference/rfm
        seqrfm_path=None if args.only_bow else seqrfm_path,
        out_dir=new_strategy_dir, model_name=args.model,
        alpha_grid=args.alpha_grid, judge_model=args.judge_model,
        max_new_tokens=args.eval_gen_tokens, cache_dir=args.cache_dir, seed=args.seed,
        completion_rfm_path=None,
        bag_of_words_path=None if args.only_seqrfm else bag_of_words_path,
        completion_diff_means_path=None,
        bag_of_words_diff_means_path=None,
    )
    return os.path.join(new_strategy_dir, "summary.csv")


def main(args):
    cfg = CONCEPT_CONFIG[args.concept_type]

    if args.concept_type == "fear":
        import fear_steer_and_evaluate as steer_and_evaluate_module
    else:
        import persona_steer_and_evaluate as steer_and_evaluate_module
    args.steer_and_evaluate_module = steer_and_evaluate_module

    with open(args.manifest_path) as f:
        manifest = json.load(f)

    all_summaries = []
    for i, entry in enumerate(manifest):
        concept_name = entry[cfg["manifest_key"]]
        dataset_path = entry["path"]
        dir_name = f"{cfg['dir_prefix']}_{safe_name(concept_name)}"
        old_strategy_dir = os.path.join(args.old_output_dir, dir_name, args.completion_strategy)
        new_strategy_dir = os.path.join(args.new_output_dir, dir_name, args.completion_strategy)

        if i == 0:
            print(f"[check] concept_type={args.concept_type}: resolving old_strategy_dir = "
                  f"{old_strategy_dir}")
            print(f"[check] {'EXISTS' if os.path.exists(old_strategy_dir) else 'DOES NOT EXIST -- fix paths before continuing'}")

        summary_path = run_one(concept_name, dataset_path, old_strategy_dir, new_strategy_dir, args)
        if summary_path:
            all_summaries.append((concept_name, summary_path))

    combined_path = os.path.join(
        args.new_output_dir, f"combined_summary_{args.concept_type}_{args.completion_strategy}_truncated.csv")
    os.makedirs(args.new_output_dir, exist_ok=True)
    header = [cfg["manifest_key"], "method", "alpha", "mean_score", "n"]
    with open(combined_path, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(header)
        for concept_name, sp in all_summaries:
            if not os.path.exists(sp):
                continue
            with open(sp) as in_f:
                reader = csv.reader(in_f)
                next(reader)  # skip header
                for row in reader:
                    writer.writerow(row)

    print(f"\nDone. Combined (bag_of_words_rfm + sequence_rfm only, truncate_at_end=True): {combined_path}")
    print("To merge with your existing 7-method results: drop the old 'bag_of_words_rfm' and "
          "'sequence_rfm' rows from your old combined_summary CSV, then concatenate this file's rows in.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--concept_type", required=True, choices=["fear", "persona"])
    p.add_argument("--manifest_path", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--old_output_dir", required=True,
                    help="root of your EXISTING {fear,persona}_experiment.py output -- reused for "
                         "trajectories.pt + dataset_with_completions.json, never written to")
    p.add_argument("--new_output_dir", required=True,
                    help="fresh directory -- never overlaps with --old_output_dir")
    p.add_argument("--completion_strategy", default="long_subsampled_anchor")
    p.add_argument("--cache_dir", default=None)
    # RFM hyperparameters -- same defaults as train_sequence_rfm.py / fear_experiment.py
    p.add_argument("--rfm_iters", type=int, default=3)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=1e-3)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--ell_tau", type=float, default=0.15)
    p.add_argument("--temporal_weight", default="uniform", choices=["uniform", "linear_late"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--fast_bow", action="store_true",
                    help="batched/vectorized kernel+gradient computation for BOTH the bag-of-words "
                         "and sequence-RFM fits (as of kernels_batched_positional.py, the positional "
                         "path is also batched now). RUN verify_fast_bow.py and "
                         "verify_fast_bow_positional.py first if you haven't already.")
    p.add_argument("--fast_bow_chunk_size", type=int, default=32)
    p.add_argument("--only_bow", action="store_true", help="skip sequence-RFM entirely (cost control)")
    p.add_argument("--only_seqrfm", action="store_true", help="skip bag-of-words-RFM entirely")
    # eval
    p.add_argument("--alpha_grid", type=float, nargs="+", required=True,
                    help="use the SAME grid as your original run for comparable results, e.g.: "
                         "0.3 0.4 0.55 0.6 0.65 0.7 0.75 0.8 0.9")
    p.add_argument("--eval_gen_tokens", type=int, default=100)
    p.add_argument("--judge_model", default="gpt-4o-mini")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.only_bow and args.only_seqrfm:
        raise ValueError("--only_bow and --only_seqrfm are mutually exclusive")

    main(args)
