"""
aggregate_results.py

Walks the topophile output directory and concatenates all
summary.csv and per_example_results.csv files into two
combined CSVs, adding 'concept' and 'strategy' columns so
you can tell which concept/run each row came from.

Usage:
    python aggregate_results.py \
        --root /data/mabel/projects/steering/update_seqrfm_v2/outputs_topophile \
        --model_tag topophile_llama31_8b_alllayers \
        --out_dir  /data/mabel/projects/steering/update_seqrfm_v2/outputs_topophile/aggregated

Outputs:
    aggregated/all_summary.csv
    aggregated/all_per_example_results.csv
"""
import argparse
import os
import pandas as pd


def collect_csvs(root, model_tag):
    summaries = []
    per_example = []

    model_dir = os.path.join(root, model_tag)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Walk: model_dir / topophile_<slug> / <strategy> / *.csv
    for concept_folder in sorted(os.listdir(model_dir)):
        concept_path = os.path.join(model_dir, concept_folder)
        if not os.path.isdir(concept_path):
            continue
        # concept_folder looks like "topophile_lisbon"
        concept_name = concept_folder.replace("topophile_", "").replace("_", " ").title()

        for strategy_folder in sorted(os.listdir(concept_path)):
            strategy_path = os.path.join(concept_path, strategy_folder)
            if not os.path.isdir(strategy_path):
                continue

            summary_path     = os.path.join(strategy_path, "summary.csv")
            per_example_path = os.path.join(strategy_path, "per_example_results.csv")

            if os.path.isfile(summary_path):
                df = pd.read_csv(summary_path)
                if "concept" not in df.columns:
                    df.insert(0, "concept", concept_name)
                if "strategy" not in df.columns:
                    df.insert(1, "strategy", strategy_folder)
                summaries.append(df)
                print(f"  [summary]     {concept_folder}/{strategy_folder}  ({len(df)} rows)")
            else:
                print(f"  [missing]     {concept_folder}/{strategy_folder}/summary.csv")

            if os.path.isfile(per_example_path):
                df = pd.read_csv(per_example_path)
                if "concept" not in df.columns:
                    df.insert(0, "concept", concept_name)
                if "strategy" not in df.columns:
                    df.insert(1, "strategy", strategy_folder)
                per_example.append(df)
                print(f"  [per_example] {concept_folder}/{strategy_folder}  ({len(df)} rows)")
            else:
                print(f"  [missing]     {concept_folder}/{strategy_folder}/per_example_results.csv")

    return summaries, per_example


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",      required=True,
                   help="outputs_topophile directory")
    p.add_argument("--model_tag", default="topophile_llama31_8b_alllayers",
                   help="model subfolder name inside --root")
    p.add_argument("--out_dir",   default=None,
                   help="where to write combined CSVs (default: <root>/aggregated)")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args.root, "aggregated")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Scanning: {os.path.join(args.root, args.model_tag)}")
    summaries, per_example = collect_csvs(args.root, args.model_tag)

    if summaries:
        combined_summary = pd.concat(summaries, ignore_index=True)
        out_path = os.path.join(out_dir, "all_summary.csv")
        combined_summary.to_csv(out_path, index=False)
        print(f"\nWrote all_summary.csv          ({len(combined_summary)} rows) -> {out_path}")
    else:
        print("\nNo summary.csv files found.")

    if per_example:
        combined_per_example = pd.concat(per_example, ignore_index=True)
        out_path = os.path.join(out_dir, "all_per_example_results.csv")
        combined_per_example.to_csv(out_path, index=False)
        print(f"Wrote all_per_example_results.csv ({len(combined_per_example)} rows) -> {out_path}")
    else:
        print("No per_example_results.csv files found.")

    # Print a quick pivot: concept x method -> best mean_score
    if summaries:
        print("\n── Best mean_score per concept × method ──")
        best = (
            combined_summary
            .groupby(["concept", "method"])["mean_score"]
            .max()
            .unstack(fill_value=float("nan"))
        )
        print(best.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()