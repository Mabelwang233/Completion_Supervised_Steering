"""
Trains sequence-RFM at EVERY layer in trajectories.pt (matching the
notebook's per-layer-direction convention), not just one.

This is the expensive step. Sequence-RFM training at a single layer was
already estimated at ~30-60 min with anchor subsampling; doing this for
every layer (~30 for an 8B model) is the dominant cost of the whole
pipeline -- roughly 15-30 hours naively. If that's too much, pass a
subset via --layers on extract_trajectories.py (e.g. every 3rd layer)
rather than changing anything here -- this script just trains whatever
layers are present in trajectories.pt.

bag_of_words=True (via --bag_of_words) trains the k_tau===1 diagnostic
variant instead -- same trajectories, same AGOP loop, position ignored
entirely. Reuses the same trajectories.pt as sequence-RFM, so this is
cheap to add on top of an existing run (no new activation extraction).

--truncate_at_end: if set, carries the FULL-RANK AGOP metric through
every intermediate iteration's kernel and only truncates to --k at the
very end -- see fit_sequence_rfm's docstring in sequence_rfm.py for the
full explanation, including a real compute-cost warning (this is NOT a
free speedup; intermediate iterations get more expensive, roughly
(n_train_examples / k)x per iteration). The payoff: with this flag on,
run ONCE at your largest desired --k (e.g. 5), and controller.py's
load_directions_for_k can slice the resulting file down to k=1/3
downstream with zero retraining -- no need for separate k=1/3/5 runs.
Without this flag (default), k=1/3/5 genuinely need separate --k runs,
since the rank-k truncation happens every iteration and feeds a
different kernel into the next round for each k.

--fast_bow: batched, vectorized kernel/gradient computation instead of
the O(n^2) Python loop -- works with EITHER --bag_of_words or the full
positional kernel (dispatches to kernels_batched_bow.py or
kernels_batched_positional.py respectively; see sequence_rfm.py's
docstring). Swaps thousands of tiny sequential GPU ops for a handful of
large batched matmuls -- the fix if you've noticed low GPU memory/
utilization during this step. RUN verify_fast_bow.py (bag-of-words
path) and/or verify_fast_bow_positional.py (positional path) FIRST --
neither was numerically checked against the reference implementation in
an environment with torch/GPU access; don't trust either for a real run
until the relevant one prints "ALL CHECKS PASSED" on your end.
"""
import argparse

import torch

from sequence_rfm import fit_sequence_rfm


def train(trajectories_path, out_path, sigma, lam, k, rfm_iters, ell_tau,
          temporal_weight, kernel_type, device, seed, bag_of_words=False,
          truncate_at_end=False, fast_bow=False, fast_bow_chunk_size=32):
    torch.manual_seed(seed)

    data = torch.load(trajectories_path)
    trajectories_per_layer = data["trajectories_per_layer"]
    labels = data["labels"]
    layers = data["layers"]
    print(f"Loaded trajectories for {len(layers)} layers, {len(labels)} examples/layer.")

    directions_per_layer = {}
    info_per_layer = {}

    tag = "bag-of-words RFM" if bag_of_words else "sequence-RFM"
    if truncate_at_end:
        tag += " [truncate_at_end=True -- this file's directions can be sliced to any k' <= k]"
    if fast_bow:
        tag += " [fast_bow=True -- batched]"
    for layer in layers:
        print(f"\n=== {tag}: layer {layer} ({layers.index(layer)+1}/{len(layers)}) ===")
        trajectories = trajectories_per_layer[layer]
        directions, info = fit_sequence_rfm(
            trajectories, labels, sigma=sigma, lam=lam, k=k, rfm_iters=rfm_iters,
            ell_tau=ell_tau, temporal_weight=temporal_weight, kernel_type=kernel_type,
            device=device, bag_of_words=bag_of_words, truncate_at_end=truncate_at_end,
            fast_bow=fast_bow, fast_bow_chunk_size=fast_bow_chunk_size,
        )
        directions_per_layer[layer] = directions.cpu()
        info_per_layer[layer] = info

    torch.save({"directions_per_layer": directions_per_layer,
                "info_per_layer": info_per_layer, "layers": layers,
                "bag_of_words": bag_of_words, "truncate_at_end": truncate_at_end,
                "fast_bow": fast_bow, "k": k}, out_path)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trajectories_path", default="../data/trajectories.pt")
    p.add_argument("--out_path", default="../outputs/sequence_rfm_directions.pt")
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=1e-3)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--rfm_iters", type=int, default=3)
    p.add_argument("--ell_tau", type=float, default=0.15)
    p.add_argument("--temporal_weight", default="uniform", choices=["uniform", "linear_late"])
    p.add_argument("--kernel_type", default="gaussian", choices=["gaussian", "laplace"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bag_of_words", action="store_true",
                    help="train the k_tau===1 diagnostic variant instead of positional sequence-RFM")
    p.add_argument("--truncate_at_end", action="store_true",
                    help="carry full-rank AGOP metric through intermediate iterations, truncate to "
                         "--k only at export -- lets one run (at max k) serve smaller k via slicing "
                         "downstream, at a real per-iteration compute cost (see docstring above)")
    p.add_argument("--fast_bow", action="store_true",
                    help="batched/vectorized kernel+gradient computation -- works with EITHER "
                         "--bag_of_words or the full positional kernel. RUN verify_fast_bow.py / "
                         "verify_fast_bow_positional.py FIRST.")
    p.add_argument("--fast_bow_chunk_size", type=int, default=32,
                    help="number of examples processed per batched chunk in --fast_bow mode -- "
                         "raise for more speed if GPU memory allows, lower if you hit OOM")
    args = p.parse_args()

    train(args.trajectories_path, args.out_path, args.sigma, args.lam, args.k,
          args.rfm_iters, args.ell_tau, args.temporal_weight, args.kernel_type,
          args.device, args.seed, bag_of_words=args.bag_of_words,
          truncate_at_end=args.truncate_at_end, fast_bow=args.fast_bow,
          fast_bow_chunk_size=args.fast_bow_chunk_size)