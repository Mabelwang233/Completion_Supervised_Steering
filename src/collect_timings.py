"""
Collect training-time timings.json files (written by timing.py) into one
table: one row per concept, plus mean / median rows across concepts.

Two kinds of columns, all in minutes:
  stage columns  -- raw wall clock of each recorded stage (includes model
                    load, except rfm/mean_difference, whose shared load is
                    baselines_model_load)
  total_<method> -- the method's training cost INCLUDING its prerequisite
                    stages (see METHOD_STAGES); never includes evaluation

A total is left blank for a concept if any of its stages wasn't recorded
(e.g. the output already existed before timing was added).

Usage:
    python collect_timings.py --output_dir ../outputs_400x2/experts_llama_8b_alllayers --prefix expert
"""
import argparse
import csv
import glob
import json
import os
from statistics import mean, median

STAGES = [
    "baselines_model_load", "rfm", "mean_difference",
    "generate_completions", "completion_rfm",
    "extract_trajectories", "bag_of_words_diff_means",
    "extract_trajectories_seqrfm", "sequence_rfm", "bag_of_words_rfm",
    "clas", "ctx_ctrl_completion_rfm", "ctx_ctrl_controller", "joint",
]

# A tuple entry means "first one recorded" -- seqRFM re-extracts trajectories
# under its own name, but falls back to the first extraction if it reused it.
METHOD_STAGES = {
    "mean_difference": ["baselines_model_load", "mean_difference"],
    "rfm": ["baselines_model_load", "rfm"],
    "completion_rfm": ["generate_completions", "completion_rfm"],
    "bag_of_words_diff_means": ["generate_completions", "extract_trajectories", "bag_of_words_diff_means"],
    "sequence_rfm": ["generate_completions", ("extract_trajectories_seqrfm", "extract_trajectories"), "sequence_rfm"],
    "bag_of_words_rfm": ["generate_completions", ("extract_trajectories_seqrfm", "extract_trajectories"), "bag_of_words_rfm"],
    # CLAS uses the vanilla rfm baseline as d_l
    "clas": ["generate_completions", "baselines_model_load", "rfm", "clas"],
    "ctx_controller": ["generate_completions", "ctx_ctrl_completion_rfm", "ctx_ctrl_controller"],
    # joint is initialized from completion_rfm
    "joint": ["generate_completions", "completion_rfm", "joint"],
}


def method_total(timings, stages):
    total = 0.0
    for stage in stages:
        options = stage if isinstance(stage, tuple) else (stage,)
        found = next((timings[s] for s in options if s in timings), None)
        if found is None:
            return None
        total += found
    return total


def fmt(seconds):
    return "" if seconds is None else round(seconds / 60, 2)


def main(args):
    paths = sorted(glob.glob(os.path.join(args.output_dir, f"{args.prefix}_*", "timings.json")))
    if not paths:
        raise SystemExit(f"no {args.prefix}_*/timings.json under {args.output_dir}")

    columns = STAGES + [f"total_{m}" for m in METHOD_STAGES]
    rows = []
    for path in paths:
        with open(path) as f:
            timings = json.load(f)
        row = {s: timings.get(s) for s in STAGES}
        row.update({f"total_{m}": method_total(timings, st) for m, st in METHOD_STAGES.items()})
        row["concept"] = os.path.basename(os.path.dirname(path))[len(args.prefix) + 1:]
        rows.append(row)

    def aggregate(fn, name):
        out = {"concept": name}
        for c in columns:
            values = [r[c] for r in rows if r[c] is not None]
            out[c] = fn(values) if values else None
        return out

    summary_rows = [aggregate(mean, "MEAN"), aggregate(median, "MEDIAN")]
    out_path = args.out_path or os.path.join(args.output_dir, "timings_summary_minutes.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept"] + columns)
        for r in rows + summary_rows:
            writer.writerow([r["concept"]] + [fmt(r[c]) for c in columns])

    n_with = {m: sum(r[f"total_{m}"] is not None for r in rows) for m in METHOD_STAGES}
    print(f"{len(rows)} concepts. Mean training time per method, minutes (incl. prerequisites):")
    for m in METHOD_STAGES:
        value = summary_rows[0][f"total_{m}"]
        print(f"  {m:<26} {fmt(value) if value is not None else '-':>8}   (n={n_with[m]})")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True, help="e.g. ../outputs_400x2/experts_llama_8b_alllayers")
    p.add_argument("--prefix", default="expert", choices=["expert", "persona", "fear"])
    p.add_argument("--out_path", default=None)
    args = p.parse_args()
    main(args)
