"""
Standalone diagnostic -- run this on your machine against the actual
trajectories.pt before changing anything in sequence_rfm.py.

Checks whether sigma=1.0 is plausible given the REAL scale of pairwise
squared distances between hidden states in your extracted trajectories.
If typical squared distances are >> 2*sigma^2, exp(-sq_dist/(2*sigma^2))
underflows to ~0 for nearly every pair, which is consistent with:
  - iter 1's suspicious 1.000 train_acc (K ~= identity, trivial fit)
  - the SVD convergence warning (gradients computed from a near-zero,
    saturated kernel are numerically degenerate)
  - iter 2/3 collapsing to exactly 0.500 (a constant/degenerate K with
    balanced labels predicts exactly zero for everyone)

Usage:
    python diagnose_kernel_scale.py --trajectories_path outputs/.../trajectories.pt --layer -1
"""
import argparse
import torch


def diagnose(trajectories_path, layer, sigma, n_pairs_sample=2000, seed=0):
    torch.manual_seed(seed)
    data = torch.load(trajectories_path)
    trajectories = data["trajectories_per_layer"][layer]
    n = len(trajectories)
    print(f"Layer {layer}: {n} trajectories, "
          f"avg {sum(t['h'].shape[0] for t in trajectories)/n:.1f} anchors each")

    # sample random pairs of (example, anchor) points -- mix of same-example
    # and cross-example pairs, since both matter for the kernel
    all_points = torch.cat([t["h"] for t in trajectories], dim=0)
    print(f"Total anchor points across all examples: {all_points.shape[0]}, dim={all_points.shape[1]}")

    idx_a = torch.randint(0, all_points.shape[0], (n_pairs_sample,))
    idx_b = torch.randint(0, all_points.shape[0], (n_pairs_sample,))
    diffs = all_points[idx_a] - all_points[idx_b]
    sq_dists = (diffs ** 2).sum(dim=-1)

    print(f"\nPairwise squared-distance stats (random sample of {n_pairs_sample} pairs):")
    print(f"  min:    {sq_dists.min().item():.2f}")
    print(f"  median: {sq_dists.median().item():.2f}")
    print(f"  mean:   {sq_dists.mean().item():.2f}")
    print(f"  max:    {sq_dists.max().item():.2f}")

    print(f"\nWith current sigma={sigma}: exp(-sq_dist / (2*sigma^2))")
    kernel_vals = torch.exp(-sq_dists / (2 * sigma ** 2))
    n_nonzero = (kernel_vals > 1e-6).sum().item()
    print(f"  fraction of sampled pairs with kernel value > 1e-6: {n_nonzero/n_pairs_sample:.4f}")
    if n_nonzero / n_pairs_sample < 0.05:
        print("  ^ WARNING: <5% of pairs have a non-negligible kernel value.")
        print("    This means K is essentially the identity matrix -- consistent")
        print("    with the train_acc=1.000 / 0.500 pattern you're seeing.")

    # median heuristic suggestion
    median_sq_dist = sq_dists.median().item()
    suggested_sigma = (median_sq_dist / 2) ** 0.5
    print(f"\nMedian-heuristic suggested sigma: {suggested_sigma:.2f}")
    print(f"  (sets sigma so that a 'typical' pair has kernel value ~= exp(-0.5) ~= 0.6,")
    print(f"   instead of ~0)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trajectories_path", required=True)
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--sigma", type=float, default=1.0, help="the sigma currently in use, to check against")
    args = p.parse_args()
    diagnose(args.trajectories_path, args.layer, args.sigma)
