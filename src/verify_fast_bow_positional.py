"""
Verifies kernels_batched_positional.py against the loop-based reference
in kernels.py / sequence_rfm.py, on small random synthetic trajectories.
Run this BEFORE trusting --fast_bow for a real (non-bag-of-words)
sequence-RFM run -- mirrors verify_fast_bow.py's role for the
bag-of-words path.

The underlying math was already checked against a from-scratch NumPy
reimplementation (in an environment without torch/GPU access) -- 20/20
randomized checks passed to floating-point precision, covering the
positional case with/without a nontrivial metric M (that prototype also
checked a bag_of_words branch as an algebra sanity check, but this
script only exercises bag_of_words=False -- kernels_batched_positional.py
itself has no bag_of_words parameter; the bag_of_words=True path is
kernels_batched_bow.py's separate scope, checked by verify_fast_bow.py).
This script re-checks the ACTUAL torch code you'll run, on your actual
device.

Usage:
    python verify_fast_bow_positional.py            # runs on cuda if available, else cpu
    python verify_fast_bow_positional.py --device cpu
"""
import argparse
import sys

import torch

from kernels import sequence_kernel_matrix, apply_metric, state_kernel
from sequence_rfm import _compute_gradients
from kernels_batched_positional import positional_kernel_matrix_batched, positional_gradients_batched


def make_random_trajectories(n, d, T_min, T_max, device, seed):
    g = torch.Generator().manual_seed(seed)
    trajs = []
    for _ in range(n):
        T_i = torch.randint(T_min, T_max + 1, (1,), generator=g).item()
        h = torch.randn(T_i, d, generator=g)
        tau_raw = torch.rand(T_i, generator=g)
        tau = torch.sort(tau_raw).values  # increasing, matches real trajectory tau
        trajs.append({"h": h.to(device), "tau": tau.to(device)})
    return trajs


def run_one_check(use_metric, seed, device, chunk_size=4,
                   n=9, d=5, sigma=1.3, ell_tau=0.2, atol=1e-5, rtol=1e-4):
    # bag_of_words is NOT a parameter of positional_kernel_matrix_batched /
    # positional_gradients_batched -- they always apply the real k_tau
    # factor, so this check is scoped to bag_of_words=False (the positional
    # case) only. The bag_of_words=True path is kernels_batched_bow.py's
    # scope, checked by verify_fast_bow.py instead -- comparing this file's
    # output against a bag_of_words=True reference would be an
    # apples-to-oranges mismatch, not a real correctness signal.
    trajs_ref = make_random_trajectories(n, d, T_min=2, T_max=6, device=device, seed=seed)
    # deep-copy h/tau since sequence_rfm._compute_gradients / kernels functions
    # don't mutate in place, but be defensive rather than assume
    trajs_batched = [{"h": t["h"].clone(), "tau": t["tau"].clone()} for t in trajs_ref]

    g = torch.Generator().manual_seed(seed + 100)
    alpha = torch.randn(n, generator=g).to(device)

    if use_metric:
        g2 = torch.Generator().manual_seed(seed + 200)
        k_rank = 2
        raw = torch.randn(d, k_rank, generator=g2)
        M_eigvecs, _ = torch.linalg.qr(raw)
        M_eigvecs = M_eigvecs.to(device)
        M_eigvals = torch.tensor([2.0, 0.5][:k_rank], device=device)
    else:
        M_eigvecs, M_eigvals = None, None

    K_ref = sequence_kernel_matrix(trajs_ref, M_eigvecs, M_eigvals, sigma, ell_tau,
                                    "gaussian", bag_of_words=False)
    K_batched = positional_kernel_matrix_batched(trajs_batched, M_eigvecs, M_eigvals, sigma, ell_tau,
                                                  kernel_type="gaussian", chunk_size=chunk_size,
                                                  device=device)
    k_ok = torch.allclose(K_ref.cpu(), K_batched.cpu(), atol=atol, rtol=rtol)
    k_max_err = (K_ref.cpu() - K_batched.cpu()).abs().max().item()

    g_ref = _compute_gradients(trajs_ref, alpha, M_eigvecs, M_eigvals, sigma, ell_tau,
                                "gaussian", bag_of_words=False)
    g_batched = positional_gradients_batched(trajs_batched, alpha, M_eigvecs, M_eigvals, sigma, ell_tau,
                                              kernel_type="gaussian", chunk_size=chunk_size,
                                              device=device)
    g_ok = all(torch.allclose(a.cpu(), b.cpu(), atol=atol, rtol=rtol) for a, b in zip(g_ref, g_batched))
    g_max_err = max((a.cpu() - b.cpu()).abs().max().item() for a, b in zip(g_ref, g_batched))

    # also check the gradients came back on the right device (this exact
    # bug bit the bag-of-words version once already -- see
    # kernels_batched_bow.py's comment about it)
    device_ok = all(str(g.device).startswith(str(device).split(":")[0]) for g in g_batched)

    label = f"use_metric={use_metric} seed={seed}"
    status = "OK" if (k_ok and g_ok and device_ok) else "FAIL"
    print(f"{label:30s}  K: {'OK' if k_ok else 'FAIL'} (max_err={k_max_err:.2e})   "
          f"grads: {'OK' if g_ok else 'FAIL'} (max_err={g_max_err:.2e})   "
          f"device: {'OK' if device_ok else 'FAIL'}   [{status}]")
    return k_ok and g_ok and device_ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"Running on device={args.device}\n")

    results = []
    for seed in range(5):
        for use_metric in (False, True):
            results.append(run_one_check(use_metric, seed, args.device))

    print()
    if all(results):
        print(f"ALL {len(results)} CHECKS PASSED")
        sys.exit(0)
    else:
        print(f"{sum(not r for r in results)} / {len(results)} CHECKS FAILED -- "
              f"do NOT trust --fast_bow for the positional case until this passes.")
        sys.exit(1)