"""
Samples a fixed subset of the 50 test questions ONCE and saves the
indices, so every window config in the ablation is evaluated on
identical questions (paired comparison, not independent samples).

Run once per dataset_with_completions.json (the test-question list is
the same generic 50 across all personas, so in practice you only need
to run this once total and reuse the same indices file everywhere --
but it's keyed by the dataset path just in case that ever changes).
"""
import argparse
import json
import random


def make_eval_subset(dataset_with_completions_path, out_path, n=20, seed=0):
    with open(dataset_with_completions_path) as f:
        dataset = json.load(f)
    test_examples = dataset["test"]

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(test_examples)), n))

    with open(out_path, "w") as f:
        json.dump({"n": n, "seed": seed, "indices": indices,
                   "questions": [test_examples[i]["question"] for i in indices]}, f, indent=2)
    print(f"Wrote {out_path}: {n} questions sampled (seed={seed}) from {len(test_examples)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_with_completions_path", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    make_eval_subset(args.dataset_with_completions_path, args.out_path, args.n, args.seed)
