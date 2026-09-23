"""
Corrects a real bug found in per_example_results.csv files generated
before the judge fix (judge.py / fear_judge.py / persona_judge.py): when
a steered generation produced a completely empty response (typically at
high alpha, where the steering push makes the very first generated
token EOS), the GPT judge scored it 1 instead of 0, fabricating a
plausible-sounding explanation for content that doesn't exist.

Pure DATA correction -- no GPU, no judge API calls. Generation is
deterministic (do_sample=False), so an empty response would regenerate
as empty again; we already know with certainty what the correct score
for "the model said nothing" is (0), so we fix it directly instead of
re-running anything.

Walks a directory tree, finds every per_example_results.csv, corrects
any row with an empty/whitespace-only response to score=0, regenerates
the corresponding summary.csv from the corrected rows, then regenerates
any combined_summary_*.csv at the top level by re-aggregating from the
(now-corrected) per-concept summary.csv files.

Usage (run once per output_dir -- personas and fears separately):
    python fix_empty_response_scores.py \
        --output_dir /path/to/outputs/personas_llama_8b_alllayers --id_field persona
    python fix_empty_response_scores.py \
        --output_dir /path/to/outputs/fears_llama_8b_alllayers --id_field fear
"""
import argparse
import csv
import glob
import os
from collections import defaultdict


def is_empty(response):
    if response is None:
        return True
    s = str(response).strip()
    return s == "" or s.lower() == "nan"


def fix_one_file(rows_path, id_field):
    with open(rows_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    n_fixed = 0
    for row in rows:
        if is_empty(row.get("response", "")):
            if row.get("score") != "0":
                n_fixed += 1
            row["score"] = "0"
            row["explanation"] = ("Empty response (model generated no text) -- corrected from a prior "
                                   "judge bug that scored empty responses as 1.")

    if n_fixed:
        with open(rows_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # regenerate summary.csv from the (corrected) rows -- same directory
    summary_path = os.path.join(os.path.dirname(rows_path), "summary.csv")
    agg = defaultdict(list)
    concept_name = None
    for row in rows:
        concept_name = row[id_field]
        agg[(row["method"], row["alpha"])].append(float(row["score"]))

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([id_field, "method", "alpha", "mean_score", "n"])
        for (method, alpha), scores in sorted(agg.items()):
            writer.writerow([concept_name, method, alpha, sum(scores) / len(scores), len(scores)])

    return n_fixed, len(rows)


def main(output_dir, id_field):
    pattern = os.path.join(output_dir, "**", "per_example_results.csv")
    files = glob.glob(pattern, recursive=True)
    print(f"Found {len(files)} per_example_results.csv files under {output_dir}")

    total_fixed = 0
    affected_files = []
    for path in sorted(files):
        n_fixed, n_total = fix_one_file(path, id_field)
        if n_fixed:
            print(f"  {path}: corrected {n_fixed}/{n_total} rows")
            affected_files.append(path)
        total_fixed += n_fixed

    print(f"\nTotal rows corrected: {total_fixed} across {len(affected_files)} files")
    if not affected_files:
        print("No empty-response corruption found -- nothing to fix here.")
        return

    # regenerate combined_summary_*.csv files by re-aggregating from the
    # (now-corrected) per-concept summary.csv files
    combined_files = glob.glob(os.path.join(output_dir, "combined_summary_*.csv"))
    for combined_path in combined_files:
        strategy_tag = os.path.basename(combined_path)[len("combined_summary_"):-len(".csv")]
        summary_paths = sorted(glob.glob(os.path.join(output_dir, f"{id_field}_*", strategy_tag, "summary.csv")))
        if not summary_paths:
            continue
        with open(combined_path, "w", newline="") as out_f:
            writer = csv.writer(out_f)
            writer.writerow([id_field, "method", "alpha", "mean_score", "n"])
            for sp in summary_paths:
                with open(sp) as in_f:
                    reader = csv.reader(in_f)
                    next(reader)
                    for row in reader:
                        writer.writerow(row)
        print(f"Regenerated {combined_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True,
                    help="e.g. outputs/personas_llama_8b_alllayers or outputs/fears_llama_8b_alllayers")
    p.add_argument("--id_field", required=True, choices=["persona", "fear"])
    args = p.parse_args()
    main(args.output_dir, args.id_field)
