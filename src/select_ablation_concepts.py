"""
select_ablation_concepts.py

Selects 10 persona + 10 fear concepts for the teacher-completion ablation
using uniform random sampling with seed=42. Reads from the existing
manifests and writes a new ablation manifest.

Usage:
    python select_ablation_concepts.py \
        --persona_manifest data_200x2/personas/manifest30.json \
        --fear_manifest    data_200x2/fears/manifest30.json \
        --out_path         teacher_ablation/ablation_manifest.json \
        --n                10 \
        --seed             42
"""
import argparse
import json
import random
import os


def select_concepts(manifest_path, n, seed, key):
    with open(manifest_path) as f:
        manifest = json.load(f)
    rng = random.Random(seed)
    selected = rng.sample(manifest, n)
    print(f"  Selected {n} from {len(manifest)} {key}s:")
    for entry in selected:
        print(f"    {entry[key]}")
    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--persona_manifest", required=True)
    p.add_argument("--fear_manifest",    required=True)
    p.add_argument("--out_path",         required=True)
    p.add_argument("--n",                type=int, default=10)
    p.add_argument("--seed",             type=int, default=42)
    args = p.parse_args()

    print(f"Selecting {args.n} personas (seed={args.seed})...")
    personas = select_concepts(args.persona_manifest, args.n, args.seed, "persona")

    print(f"\nSelecting {args.n} fears (seed={args.seed})...")
    fears = select_concepts(args.fear_manifest, args.n, args.seed, "fear")

    out = {
        "seed": args.seed,
        "n_per_category": args.n,
        "personas": personas,
        "fears": fears,
    }

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out_path}")
    print(f"Total concepts: {len(personas)} personas + {len(fears)} fears = {len(personas)+len(fears)}")


if __name__ == "__main__":
    main()
