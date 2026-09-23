"""
persona_experiment_seqrfm.py

Runs ONLY sequence-RFM and bag-of-words RFM for each persona in the
manifest, both with truncate_at_end=True and fast_bow=True.

Reuses existing artifacts where they already exist (completions,
trajectories) so this can be dropped on top of an existing
long_subsampled_anchor run without regenerating anything.

Steps per concept:
  [1/5] generate completions      -- skipped if dataset_with_completions.json exists
  [2/5] extract trajectories      -- skipped if trajectories.pt exists
  [3/5] train sequence-RFM        -- truncate_at_end=True, fast_bow=True
  [4/5] train bag-of-words RFM    -- same flags, bag_of_words=True
  [5/5] steer + evaluate          -- sequence_rfm + bag_of_words_rfm only
        (trajectories.pt deleted after step 4 to free disk)

Output layout (mirrors persona_experiment.py):
  <output_dir>/persona_<name>/<completion_strategy>/
      sequence_rfm_directions.pt
      bag_of_words_rfm_directions.pt
      per_example_results.csv      (both methods combined)
      summary.csv                  (both methods combined)
  <output_dir>/combined_summary_seqrfm_<completion_strategy>.csv
"""
import argparse
import csv
import json
import os

import generate_completions
import extract_trajectories
import train_sequence_rfm
import persona_steer_and_evaluate


def run_one_concept(persona, dataset_path, out_dir, args):
    os.makedirs(out_dir, exist_ok=True)
    strategy_dir = os.path.join(out_dir, args.completion_strategy)
    os.makedirs(strategy_dir, exist_ok=True)

    dataset_with_completions_path = os.path.join(strategy_dir, "dataset_with_completions.json")
    trajectories_path             = os.path.join(strategy_dir, "trajectories.pt")
    seqrfm_path                   = os.path.join(strategy_dir, "sequence_rfm_directions.pt")
    bag_of_words_path             = os.path.join(strategy_dir, "bag_of_words_rfm_directions.pt")

    print(f"\n{'='*60}")
    print(f"PERSONA: {persona}  |  STRATEGY: {args.completion_strategy}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Step 1: generate completions (reuse if already present)
    # ------------------------------------------------------------------
    print(f"\n[1/5] generate completions (max {args.train_completion_tokens} tokens)")
    if not os.path.exists(dataset_with_completions_path):
        generate_completions.generate_completions(
            dataset_path, dataset_with_completions_path, args.model,
            cache_dir=args.cache_dir, max_new_tokens=args.train_completion_tokens,
            seed=args.seed, train_prompt_suffix=args.train_prompt_suffix,
        )
    else:
        print(f"  {dataset_with_completions_path} exists, skipping.")

    # ------------------------------------------------------------------
    # Step 2: extract trajectories (reuse if already present)
    # ------------------------------------------------------------------
    print(f"\n[2/5] extract trajectories (n_anchors={args.n_anchors})")
    if not os.path.exists(trajectories_path):
        extract_trajectories.extract_trajectories(
            dataset_with_completions_path, trajectories_path, args.model, args.layers,
            n_anchors=args.n_anchors, cache_dir=args.cache_dir, seed=args.seed,
        )
    else:
        print(f"  {trajectories_path} exists, skipping.")

    # ------------------------------------------------------------------
    # Step 3: sequence-RFM  (truncate_at_end=True, fast_bow=True)
    # ------------------------------------------------------------------
    print(f"\n[3/5] train sequence-RFM "
          f"(truncate_at_end=True, fast_bow=True, chunk={args.fast_bow_chunk_size})")
    if not os.path.exists(seqrfm_path):
        train_sequence_rfm.train(
            trajectories_path, seqrfm_path,
            sigma=args.sigma, lam=args.lam, k=args.k,
            rfm_iters=args.rfm_iters, ell_tau=args.ell_tau,
            temporal_weight=args.temporal_weight,
            kernel_type="gaussian", device="cuda", seed=args.seed,
            bag_of_words=False,
            truncate_at_end=True,
            fast_bow=True,
            fast_bow_chunk_size=args.fast_bow_chunk_size,
        )
    else:
        print(f"  {seqrfm_path} exists, skipping.")

    # ------------------------------------------------------------------
    # Step 4: bag-of-words RFM  (same flags, bag_of_words=True)
    # ------------------------------------------------------------------
    print(f"\n[4/5] train bag-of-words RFM "
          f"(truncate_at_end=True, fast_bow=True, chunk={args.fast_bow_chunk_size})")
    if not os.path.exists(bag_of_words_path):
        train_sequence_rfm.train(
            trajectories_path, bag_of_words_path,
            sigma=args.sigma, lam=args.lam, k=args.k,
            rfm_iters=args.rfm_iters, ell_tau=args.ell_tau,
            temporal_weight=args.temporal_weight,
            kernel_type="gaussian", device="cuda", seed=args.seed,
            bag_of_words=True,
            truncate_at_end=True,
            fast_bow=True,
            fast_bow_chunk_size=args.fast_bow_chunk_size,
        )
    else:
        print(f"  {bag_of_words_path} exists, skipping.")

    # trajectories.pt no longer needed — free the disk space
    if os.path.exists(trajectories_path):
        os.remove(trajectories_path)
        print(f"  Deleted {trajectories_path} (~19 GB freed)")

    # ------------------------------------------------------------------
    # Step 5: steer + evaluate  (sequence_rfm + bag_of_words_rfm only)
    # Write to a SEPARATE temp subdirectory first, then APPEND those
    # rows into the existing summary.csv / per_example_results.csv so
    # we never overwrite what the main run already produced.
    # ------------------------------------------------------------------
    print(f"\n[5/5] steer + evaluate "
          f"(sequence_rfm + bag_of_words_rfm, alpha sweep, "
          f"max {args.eval_gen_tokens} tokens)")

    # Temporary output dir — results land here first
    seqrfm_tmp_dir = os.path.join(strategy_dir, "_seqrfm_tmp")
    os.makedirs(seqrfm_tmp_dir, exist_ok=True)

    persona_steer_and_evaluate.run_evaluation(
        dataset_path,
        baseline_path=None,
        seqrfm_path=seqrfm_path,
        out_dir=seqrfm_tmp_dir,
        model_name=args.model,
        alpha_grid=args.alpha_grid,
        judge_model=args.judge_model,
        max_new_tokens=args.eval_gen_tokens,
        cache_dir=args.cache_dir,
        seed=args.seed,
        completion_rfm_path=None,
        bag_of_words_path=bag_of_words_path,
        completion_diff_means_path=None,
        bag_of_words_diff_means_path=None,
        seqrfm_method_name="sequence_rfm",
    )

    # Append new rows into the existing summary.csv / per_example_results.csv
    import csv, shutil
    for filename in ("summary.csv", "per_example_results.csv"):
        src  = os.path.join(seqrfm_tmp_dir, filename)
        dest = os.path.join(strategy_dir,   filename)
        if not os.path.exists(src):
            print(f"  WARNING: {src} not found, skipping merge for {filename}")
            continue
        with open(src, newline="") as in_f:
            reader   = csv.reader(in_f)
            header   = next(reader)
            new_rows = list(reader)
        if os.path.exists(dest):
            # Existing file — append data rows only, no duplicate header
            with open(dest, "a", newline="") as out_f:
                csv.writer(out_f).writerows(new_rows)
            print(f"  Appended {len(new_rows)} rows -> {dest}")
        else:
            # No prior file — write fresh with header
            with open(dest, "w", newline="") as out_f:
                w = csv.writer(out_f)
                w.writerow(header)
                w.writerows(new_rows)
            print(f"  Wrote {dest} (no prior file found)")

    shutil.rmtree(seqrfm_tmp_dir)
    print(f"  Removed temp dir {seqrfm_tmp_dir}")


def main(args):
    with open(args.manifest_path) as f:
        manifest = json.load(f)

    all_summaries = []
    for entry in manifest:
        persona = entry["persona"]
        dataset_path = entry["path"]
        safe_name = persona.lower().replace(" ", "_")
        out_dir = os.path.join(args.output_dir, f"persona_{safe_name}")
        run_one_concept(persona, dataset_path, out_dir, args)
        all_summaries.append(os.path.join(out_dir, args.completion_strategy, "summary.csv"))

    # Combine all per-concept summaries
    combined_path = os.path.join(
        args.output_dir,
        f"combined_summary_seqrfm_{args.completion_strategy}.csv"
    )
    with open(combined_path, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["persona", "method", "alpha", "mean_score", "n"])
        for summary_path in all_summaries:
            if not os.path.exists(summary_path):
                continue
            with open(summary_path) as in_f:
                reader = csv.reader(in_f)
                next(reader)  # skip header
                # only write seqrfm / bag_of_words_rfm rows
                for row in reader:
                    if row[1] in ("sequence_rfm", "bag_of_words_rfm"):
                        writer.writerow(row)

    print(f"\nAll concepts done.")
    print(f"Combined summary (seqrfm + bow): {combined_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest_path",           required=True)
    p.add_argument("--model",                   required=True)
    p.add_argument("--output_dir",              required=True)
    p.add_argument("--cache_dir",               default=None)
    p.add_argument("--layers",                  nargs="+", default=None,
                   help="explicit layer indices, or omit for all layers")
    p.add_argument("--rfm_iters",               type=int,   default=3)
    p.add_argument("--sigma",                   type=float, default=1.0)
    p.add_argument("--lam",                     type=float, default=1e-3)
    p.add_argument("--k",                       type=int,   default=1)
    p.add_argument("--n_anchors",               type=int,   default=100)
    p.add_argument("--ell_tau",                 type=float, default=0.15)
    p.add_argument("--temporal_weight",         default="uniform",
                   choices=["uniform", "linear_late"])
    p.add_argument("--fast_bow_chunk_size",     type=int,   default=32,
                   help="chunk size for fast_bow batched computation -- "
                        "raise if GPU memory allows, lower on OOM")
    p.add_argument("--alpha_grid",              type=float, nargs="+", required=True)
    p.add_argument("--completion_strategy",     required=True,
                   choices=["short_instructed_cut", "long_subsampled_anchor"])
    p.add_argument("--train_prompt_suffix",     default=None)
    p.add_argument("--train_completion_tokens", type=int,   default=100)
    p.add_argument("--eval_gen_tokens",         type=int,   default=100)
    p.add_argument("--judge_model",             default="gpt-4o-mini")
    p.add_argument("--seed",                    type=int,   default=42)
    args = p.parse_args()

    main(args)