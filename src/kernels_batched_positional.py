"""
Batched (vectorized) POSITIONAL sequence-RFM kernel matrix + gradient
computation -- extends kernels_batched_bow.py's fast path to the full
k_tau-weighted case (kernels.py's sequence_kernel_matrix /
sequence_rfm.py's _compute_gradients with bag_of_words=False), instead
of only the k_tau===1 special case.

Reuses kernels_batched_bow.py's padding/whitening/pairwise-kernel
helpers unchanged (imported, not duplicated) -- the only new work here
is adding the k_tau factor. See that file's module docstring for the
padding/whitening/chunking rationale; this docstring only covers what's
different.

*** CORRECTNESS: the underlying math (this file's batched formula vs.
kernels.py/sequence_rfm.py's loop-based reference) was verified against
a from-scratch NumPy reimplementation of both sides, in an environment
without torch/GPU access -- 20/20 randomized checks passed to
floating-point precision (~1e-16), covering bag_of_words True/False and
with/without a nontrivial metric M. That confirms the ALGEBRA is
correct. It does NOT confirm this actual torch file, on your actual
GPU, with your actual dtypes -- run verify_fast_bow_positional.py
(same directory) on your end BEFORE trusting this for a real training
run, same as kernels_batched_bow.py's own fast_bow warning already
asks for the bag-of-words case. ***

MATH DERIVATION (positional: general k_tau, not assumed === 1)

Kernel matrix:
    K_ij = (1/(T_i T_j)) * sum_{t,s} k_tau(tau_i,t, tau_j,s) * k_M(h_i,t, h_j,s)

k_state = k_M(h_i,t, h_j,s) is computed exactly as in the bag-of-words
case (same whitening trick, same _pairwise_kstate_chunk). The only
addition is an elementwise k_tau factor:
    k_tau(tau_i,t, tau_j,s) = exp(-(tau_i,t - tau_j,s)^2 / (2*ell_tau^2)),
    thresholded to 0 below sparsity_eps (matches kernels.k_tau_rbf exactly).

combined_{(i,t),(j,s)} = k_tau_{(i,t),(j,s)} * k_state_{(i,t),(j,s)}

Padded positions get tau=0 (same as a real tau=0 anchor), which could
in principle give k_tau a spurious nonzero value there -- but k_state
is ALREADY zero at any padded pair (via the mask multiply inside
_pairwise_kstate_chunk), so combined = k_tau * 0 = 0 regardless. No
separate masking needed for k_tau.

Gradients: identical derivation to kernels_batched_bow.py's, with
`weighted = k_state * w_flat` there replaced by
`weighted = k_tau * k_state * w_flat` here -- everything downstream
(C_it, second_term, the final per-token gradient assembly) is
unchanged, since it was already written generically in terms of
`weighted`.
"""
import torch

from kernels_batched_bow import (
    _pad_trajectories,
    _whiten,
    _metric_apply,
    _pairwise_kstate_chunk,
)


def _pad_tau(trajectories, T_max, device):
    """Returns tau_pad (n, T_max), padded with 0 (safe -- see module
    docstring for why padded positions never spuriously contribute)."""
    n = len(trajectories)
    tau_pad = torch.zeros(n, T_max, device=device)
    for i, t in enumerate(trajectories):
        T_i = t["h"].shape[0]
        tau_pad[i, :T_i] = t["tau"].to(device)
    return tau_pad


def _k_tau_chunk(tau_chunk, tau_flat, ell_tau, sparsity_eps=1e-4):
    """tau_chunk: (c*T_max,), tau_flat: (N,). Returns (c*T_max, N),
    matching kernels.k_tau_rbf's formula and sparsity thresholding
    exactly."""
    diff = tau_chunk.unsqueeze(1) - tau_flat.unsqueeze(0)
    w = torch.exp(-(diff ** 2) / (2 * ell_tau ** 2))
    w = torch.where(w < sparsity_eps, torch.zeros_like(w), w)
    return w


def positional_kernel_matrix_batched(trajectories, M_eigvecs, M_eigvals, sigma, ell_tau,
                                      kernel_type="gaussian", chunk_size=32, device="cuda",
                                      sparsity_eps=1e-4):
    """Drop-in replacement for kernels.sequence_kernel_matrix(...,
    bag_of_words=False) -- same (n, n) output, computed via batched
    matmuls instead of an O(n^2) Python loop."""
    n = len(trajectories)
    H_pad, mask, T_list = _pad_trajectories(trajectories, device)
    T_max = H_pad.shape[1]
    tau_pad = _pad_tau(trajectories, T_max, device)

    phi = _whiten(H_pad, M_eigvecs, M_eigvals)          # (n, T_max, r)
    phi_flat = phi.reshape(n * T_max, -1)
    mask_flat = mask.reshape(n * T_max)
    tau_flat = tau_pad.reshape(n * T_max)

    K = torch.zeros(n, n, device=device)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        c = end - start
        phi_chunk = phi[start:end].reshape(c * T_max, -1)
        mask_chunk_flat = mask[start:end].reshape(c * T_max)
        tau_chunk = tau_pad[start:end].reshape(c * T_max)

        k_state = _pairwise_kstate_chunk(phi_chunk, phi_flat, mask_chunk_flat,
                                          mask_flat, sigma, kernel_type)     # (c*T_max, N)
        k_tau = _k_tau_chunk(tau_chunk, tau_flat, ell_tau, sparsity_eps)    # (c*T_max, N)
        combined = k_tau * k_state

        summed = combined.reshape(c, T_max, n, T_max).sum(dim=(1, 3))       # (c, n)

        T_i_chunk = T_list[start:end].float()
        T_j_all = T_list.float()
        K[start:end, :] = summed / (T_i_chunk.unsqueeze(1) * T_j_all.unsqueeze(0))

    return K.cpu()


def positional_gradients_batched(trajectories, alpha, M_eigvecs, M_eigvals, sigma, ell_tau,
                                  kernel_type="gaussian", chunk_size=32, device="cuda",
                                  sparsity_eps=1e-4):
    """Drop-in replacement for sequence_rfm._compute_gradients(...,
    bag_of_words=False) -- same return type (list of n (T_i, d)
    tensors), computed via batched matmuls.

    NOTE: returned tensors stay on `device` (NOT moved to .cpu()) --
    matches kernels_batched_bow.py's bow_gradients_batched contract
    exactly (see its docstring for why: the caller in sequence_rfm.py
    immediately does a device-resident elementwise multiply on the
    result with no .to(device) of its own)."""
    n = len(trajectories)
    H_pad, mask, T_list = _pad_trajectories(trajectories, device)
    T_max = H_pad.shape[1]
    d = H_pad.shape[-1]
    tau_pad = _pad_tau(trajectories, T_max, device)

    phi = _whiten(H_pad, M_eigvecs, M_eigvals)
    phi_flat = phi.reshape(n * T_max, -1)
    mask_flat = mask.reshape(n * T_max)
    tau_flat = tau_pad.reshape(n * T_max)

    Mh = _metric_apply(H_pad, M_eigvecs, M_eigvals)     # (n, T_max, d)
    Mh_flat = (Mh * mask.unsqueeze(-1)).reshape(n * T_max, d)

    alpha_dev = alpha.to(device).float()
    T_list_f = T_list.float()
    w_per_example = (alpha_dev / T_list_f)                      # (n,)
    w_flat = w_per_example.unsqueeze(1).expand(n, T_max).reshape(n * T_max)  # (N,)

    grads_flat = torch.zeros(n, T_max, d, device=device)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        c = end - start
        phi_chunk = phi[start:end].reshape(c * T_max, -1)
        mask_chunk_flat = mask[start:end].reshape(c * T_max)
        tau_chunk = tau_pad[start:end].reshape(c * T_max)

        k_state = _pairwise_kstate_chunk(phi_chunk, phi_flat, mask_chunk_flat,
                                          mask_flat, sigma, kernel_type)     # (c*T_max, N)
        k_tau = _k_tau_chunk(tau_chunk, tau_flat, ell_tau, sparsity_eps)    # (c*T_max, N)
        weighted = k_tau * k_state * w_flat.unsqueeze(0)                    # (c*T_max, N)

        C_it = weighted.sum(dim=1)                                          # (c*T_max,)
        second_term = weighted @ Mh_flat                                    # (c*T_max, d)

        Mh_chunk = Mh[start:end].reshape(c * T_max, d)
        g_flat_chunk = Mh_chunk * C_it.unsqueeze(-1) - second_term          # (c*T_max, d)
        g_flat_chunk = g_flat_chunk * mask_chunk_flat.unsqueeze(-1)

        g_chunk = g_flat_chunk.reshape(c, T_max, d)
        T_i_chunk = T_list[start:end].float().unsqueeze(-1).unsqueeze(-1)   # (c,1,1)
        g_chunk = -g_chunk / (sigma ** 2 * T_i_chunk)

        grads_flat[start:end] = g_chunk

    grads = []
    for i in range(n):
        T_i = int(T_list[i].item())
        grads.append(grads_flat[i, :T_i])
    return grads
