"""
compare_teacher_vs_self.py

Merges self-completion and teacher-completion results into a paired
comparison table.

Self-completion scores are read from pre-computed best-score CSVs stored
under /data/mabel/projects/steering/update_seqrfm_v2/all_combined:
    <all_combined_dir>/<model>_<category>_best.csv
    Columns: concept, method, score, alpha   (already best-alpha-selected)

Teacher-completion scores are read from the teacher ablation output:
    <teacher_summary>
    Columns: concept/persona/fear, method, alpha, mean_score, n
    (best alpha selected here before comparison)

Outputs:
    <out_path>  -- per-concept paired CSV
    Printed terminal table: mean self, mean teacher, mean delta, 95% CI

Usage:
    python compare_teacher_vs_self.py \\
        --all_combined_dir  /data/mabel/projects/steering/update_seqrfm_v2/all_combined \\
        --teacher_summary   teacher_ablation/outputs/llama/combined_summary_teacher_persona.csv \\
        --category          persona \\
        --model_tag         llama \\
        --out_path          teacher_ablation/comparison_persona_llama.csv
"""
import argparse
import csv
import math
import os
from collections import defaultdict


def load_self_scores(all_combined_dir, model_tag, category):
    """
    Load pre-computed best scores from all_combined/<model>_<category>_best.csv.
    Returns {(concept, method): score}
    """
    path = os.path.join(all_combined_dir, f"{model_tag}_{category}_best.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Self-completion best scores not found: {path}\n"
            f"Expected format: concept, method, score, alpha"
        )
    scores = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores[(row["concept"], row["method"])] = float(row["score"])
    print(f"  Loaded {len(scores)} self-completion scores from {path}")
    return scores


def load_teacher_scores(teacher_summary_path, concept_col):
    """
    Load teacher summary CSV, return {(concept, method): best_mean_score}
    (selects best alpha per concept x method).
    """
    best = defaultdict(float)
    with open(teacher_summary_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            concept = row[concept_col]
            method  = row["method"]
            score   = float(row["mean_score"])
            key = (concept, method)
            if score > best[key]:
                best[key] = score
    print(f"  Loaded {len(best)} teacher scores from {teacher_summary_path}")
    return dict(best)


def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def stdev(vals):
    if len(vals) < 2:
        return float("nan")
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def t_ci_95(vals):
    n = len(vals)
    if n < 2:
        return float("nan")
    t_table = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,
                6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,
                11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,
                16:2.120,17:2.110,18:2.101,19:2.093,20:2.086}
    t_crit = t_table.get(n - 1, 1.960)
    return t_crit * stdev(vals) / math.sqrt(n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all_combined_dir", required=True,
                   help="directory containing <model>_<category>_best.csv files "
                        "(e.g. /data/.../all_combined)")
    p.add_argument("--teacher_summary",  required=True,
                   help="combined_summary_teacher_<category>.csv from run_teacher_ablation.py")
    p.add_argument("--category",         required=True, choices=["persona", "fear"])
    p.add_argument("--model_tag",        required=True,
                   help="model tag matching the filename prefix (e.g. 'llama', 'qwen')")
    p.add_argument("--out_path",         required=True)
    args = p.parse_args()

    # concept column name may be "persona", "fear", or "concept" depending on the script
    # that wrote the teacher summary -- try all three
    concept_col = args.category
    self_scores    = load_self_scores(args.all_combined_dir, args.model_tag, args.category)
    teacher_scores = load_teacher_scores(args.teacher_summary, concept_col)

    # If teacher summary used "concept" column instead of "persona"/"fear"
    if not teacher_scores:
        teacher_scores = load_teacher_scores(args.teacher_summary, "concept")

    # Find paired (concept, method) keys present in both
    all_keys = set(self_scores.keys()) & set(teacher_scores.keys())
    if not all_keys:
        print("WARNING: no overlapping (concept, method) pairs found between self and teacher results.")
        print(f"  Self keys (sample):    {list(self_scores.keys())[:3]}")
        print(f"  Teacher keys (sample): {list(teacher_scores.keys())[:3]}")
        return

    methods  = sorted({m for _, m in all_keys})
    concepts = sorted({c for c, _ in all_keys})
    print(f"\n  Paired: {len(concepts)} concepts x {len(methods)} methods = {len(all_keys)} pairs")

    rows = []
    for concept in concepts:
        for method in methods:
            key = (concept, method)
            if key not in self_scores or key not in teacher_scores:
                continue
            s = self_scores[key]
            t = teacher_scores[key]
            rows.append({
                "concept":       concept,
                "method":        method,
                "model":         args.model_tag,
                "category":      args.category,
                "self_score":    round(s, 4),
                "teacher_score": round(t, 4),
                "delta":         round(t - s, 4),
            })

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    fieldnames = ["concept", "method", "model", "category",
                  "self_score", "teacher_score", "delta"]
    with open(args.out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.out_path}  ({len(rows)} rows)")

    # Terminal summary table
    print(f"\n{'='*72}")
    print(f"Paired comparison: self vs teacher  |  {args.category}  |  {args.model_tag}")
    print(f"{'='*72}")
    print(f"{'Method':<35} {'Self':>6} {'Teacher':>8} {'ΔMean':>7} {'95%CI':>9} {'↑':>4} {'=':>4} {'↓':>4}")
    print("-" * 72)
    for method in methods:
        method_rows = [r for r in rows if r["method"] == method]
        if not method_rows:
            continue
        self_vals    = [r["self_score"]    for r in method_rows]
        teacher_vals = [r["teacher_score"] for r in method_rows]
        deltas       = [r["delta"]         for r in method_rows]
        m_delta = mean(deltas)
        ci      = t_ci_95(deltas)
        n_up    = sum(1 for d in deltas if d >  0.01)
        n_same  = sum(1 for d in deltas if abs(d) <= 0.01)
        n_down  = sum(1 for d in deltas if d < -0.01)
        print(f"{method:<35} {mean(self_vals):>6.3f} {mean(teacher_vals):>8.3f} "
              f"{m_delta:>+7.3f} {ci:>9.3f} {n_up:>4} {n_same:>4} {n_down:>4}")
    print(f"\n(↑/=/↓ = improved/unchanged/worsened by >0.01, n={len(concepts)} concepts)")


if __name__ == "__main__":
    main()
