"""
Rank layers by how linearly separable the concept is, using a CHEAP
per-layer logistic probe on prompt-level (last-token) activations --
no completions, no trajectories, no sequence-RFM needed. This is meant
to run BEFORE extract_trajectories.py / train_sequence_rfm.py, so you
can restrict the expensive sequence-RFM step to just the top-K layers
this identifies, instead of guessing or paying for all layers.

Rationale: sequence-RFM training was the dominant cost in this project
by a wide margin (see earlier discussion -- ~30-60 min/layer, scaling
to many hours across a full model). This script's entire cost is one
forward pass over the training prompts (shared with what baselines.py
already needs) plus a handful of cheap sklearn LogisticRegression fits
per layer -- on the order of a minute or two, not hours.

This does NOT replace running sequence-RFM -- it's a cheap proxy signal
for which layers are worth the expensive method's time. Concepts can in
principle be more separable in a trajectory-aware way at a layer that
scores lower here, so treat this as a prior to allocate budget with,
not a guarantee.

Usage:
    python select_layers.py --dataset_path ../data/shakespeare_dataset.json \
        --model meta-llama/Llama-3.1-8B-Instruct --top_k 5
"""
import argparse
import json
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from utils import load_model
from direction_utils import get_hidden_states


def rank_layers(dataset_path, model_name, layers=None, val_frac=0.2,
                 forward_batch_size=8, cache_dir=None, seed=0):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    with open(dataset_path) as f:
        dataset = json.load(f)
    prompts = [ex["prompt"] for ex in dataset["train"]]
    labels = np.array([ex["label"] for ex in dataset["train"]])

    language_model, tokenizer = load_model(model_name, cache_dir=cache_dir)
    num_layers = len(language_model.model.layers)
    if layers is None:
        layers = list(range(-1, -num_layers, -1))  # matches the notebook default

    # held-out split for probe validation -- same idea as the internal
    # RFM validation split, just a plain random split here since this
    # only needs to rank layers, not tune real hyperparameters
    n = len(prompts)
    idx = rng.permutation(n)
    n_val = int(round(n * val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_prompts = [prompts[i] for i in train_idx]
    val_prompts = [prompts[i] for i in val_idx]
    train_y, val_y = labels[train_idx], labels[val_idx]

    print(f"Extracting last-token activations at {len(layers)} layers "
          f"({len(train_prompts)} train / {len(val_prompts)} val prompts)...")
    train_hidden = get_hidden_states(train_prompts, language_model, tokenizer, layers, forward_batch_size)
    val_hidden = get_hidden_states(val_prompts, language_model, tokenizer, layers, forward_batch_size)

    results = []
    for layer in layers:
        X_train = train_hidden[layer].float().cpu().numpy()
        X_val = val_hidden[layer].float().cpu().numpy()

        best_auc = -1.0
        for C in [1000, 100, 10, 1, 1e-1, 1e-2]:
            clf = LogisticRegression(max_iter=1000, C=C)
            clf.fit(X_train, train_y)
            val_probs = clf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(val_y, val_probs)
            best_auc = max(best_auc, auc)

        results.append({"layer": layer, "val_auc": best_auc})
        print(f"  layer {layer:>4}: val_auc={best_auc:.4f}")

    results.sort(key=lambda r: r["val_auc"], reverse=True)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--layers", nargs="+", type=int, default=None,
                    help="explicit layer indices to rank, or omit for all layers")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--forward_batch_size", type=int, default=8)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--out_path", default=None, help="optional: save full ranking as json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    results = rank_layers(args.dataset_path, args.model, args.layers, args.val_frac,
                           args.forward_batch_size, args.cache_dir, args.seed)

    top_layers = [r["layer"] for r in results[:args.top_k]]
    print(f"\nTop {args.top_k} layers by validation AUC: {top_layers}")
    print(f"Use these directly as --layers for extract_trajectories.py / train_sequence_rfm.py:")
    print(f"  --layers {' '.join(str(l) for l in top_layers)}")

    if args.out_path:
        with open(args.out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull ranking saved to {args.out_path}")
