"""
Bag-of-words diff-means: mean-pool each example's anchor activations into
one vector (same pooling bag_of_words_rfm's kernel implicitly performs),
then diff-means over labels -- instead of RFM/AGOP.

    bag-of-words RFM         : all anchor tokens, RFM/AGOP (k_tau===1)
    bag-of-words diff-means  : all anchor tokens, diff-means            <- this file

This is the closest thing in our codebase to the actual recipe used by
Persona Vectors (Chen et al. 2025): mean over response/completion
tokens, then difference in means. Needs no model reload and no new
activation extraction -- reuses trajectories.pt directly, so this is
essentially free to compute.
"""
import argparse

import torch


def train_bag_of_words_diff_means(trajectories_path, out_path, seed=0):
    torch.manual_seed(seed)

    data = torch.load(trajectories_path)
    trajectories_per_layer = data["trajectories_per_layer"]
    labels = data["labels"]  # (n,) in {-1, +1}
    layers = data["layers"]

    print(f"Training bag-of-words diff-means at {len(layers)} layers, {len(labels)} examples/layer...")

    directions_per_layer = {}
    for layer in layers:
        trajectories = trajectories_per_layer[layer]
        # mean-pool each example's anchors into one (d,) vector
        pooled = torch.stack([t["h"].mean(dim=0) for t in trajectories], dim=0)  # (n, d)

        pos_mask = labels == 1
        neg_mask = labels == -1
        direction = pooled[pos_mask].mean(dim=0) - pooled[neg_mask].mean(dim=0)  # (d,)
        direction = direction / direction.norm().clamp(min=1e-8)

        directions_per_layer[layer] = direction.unsqueeze(1)  # (d, 1), matches bag_of_words_rfm's shape

    torch.save({"directions_per_layer": directions_per_layer, "layers": layers}, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trajectories_path", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    train_bag_of_words_diff_means(args.trajectories_path, args.out_path, args.seed)
