"""
Batched (vectorized) bag-of-words RFM kernel matrix + gradient
computation -- an opt-in fast path for sequence_rfm.py's bag_of_words
mode ONLY (kernels.py's positional path, with k_tau, is untouched --
out of scope here, and higher-risk to batch given the extra k_tau
factor).

WHY THIS EXISTS: kernels.py's sequence_kernel_matrix and
sequence_rfm.py's _compute_gradients both do an O(n^2) PYTHON loop over
example pairs (i, j), issuing one small GPU op per pair -- for n~400
that's up to ~80,000 tiny sequential kernel launches. Each op is small
(hence low reported GPU memory use), but wall-clock time is dominated
by Python-loop + kernel-launch overhead between them, not by actual
compute -- a classic GPU-underutilization pattern. Since bag_of_words
mode drops the k_tau factor entirely (position ignored), the whole
computation reduces to pairwise TOKEN-level kernel evaluations that can
be vectorized via standard matmul tricks, cutting the O(n^2) Python
loop down to O(n / chunk_size) iterations, each a single large,
GPU-efficient batched op.

*** CORRECTNESS: this file has NOT been numerically verified against
kernels.py's reference (loop-based) implementation -- torch/GPU aren't
available in the sandbox this was written in. Run verify_fast_bow.py
(same directory) BEFORE trusting this for a real training run -- it
checks this file's outputs against kernels.py's on a small synthetic
example and will tell you immediately if something's wrong. ***

MATH DERIVATION (bag-of-words: k_tau === 1 throughout)

Kernel matrix:
    K_ij = (1 / (T_i T_j)) * sum_{t,s} k_M(h_i,t, h_j,s)
    k_M(a,b) = exp(-(a-b)^T M (a-b) / (2*sigma^2))     [gaussian]

With M = V diag(eigvals) V^T (low-rank, M_eigvecs=V, M_eigvals=eigvals),
define the "whitening" map phi(x) = diag(sqrt(eigvals)) V^T x. Then
    (a-b)^T M (a-b) = ||phi(a) - phi(b)||^2
so distances under M reduce to ordinary Euclidean distance after one
linear transform -- computable for ALL token pairs at once via the
standard ||a||^2 + ||b||^2 - 2*a.b matmul trick, instead of the
elementwise-broadcast-per-pair approach kernels.py's state_kernel uses
(correct, but only ever invoked one example-pair at a time here).

Gradients (pdf Section 5, bag_of_words: k_tau === 1):
    g_i,t = -(1/(sigma^2 T_i)) * sum_j (alpha_j/T_j) * sum_s
                k_M(h_i,t, h_j,s) * M(h_i,t - h_j,s)

M is linear, so M(a-b) = Ma - Mb. Let w_{j,s} = alpha_j/T_j (zero at
padded s), c_{(i,t),(j,s)} = w_{j,s} * k_M(h_i,t, h_j,s), and
Mh = the metric applied to every token. Then:
    sum_{j,s} c_{(i,t),(j,s)} * (M h_i,t - M h_j,s)
  = (M h_i,t) * C_{i,t}  -  sum_{j,s} c_{(i,t),(j,s)} * (M h_j,s)
where C_{i,t} = sum_{j,s} c_{(i,t),(j,s)} (a per-token scalar) and the
second term is a single matmul: (tokens-in-chunk x N) @ (N x d) Mh
matrix -> (tokens-in-chunk x d). Both terms are big, GPU-efficient
batched ops instead of a per-(i,j)-pair Python loop.
"""
import torch


def _pad_trajectories(trajectories, device):
    """Returns H_pad (n, T_max, d), mask (n, T_max) bool, T_list (n,) long."""
    n = len(trajectories)
    d = trajectories[0]["h"].shape[1]
    T_list = torch.tensor([t["h"].shape[0] for t in trajectories], device=device)
    T_max = int(T_list.max().item())

    H_pad = torch.zeros(n, T_max, d, device=device)
    mask = torch.zeros(n, T_max, dtype=torch.bool, device=device)
    for i, t in enumerate(trajectories):
        T_i = t["h"].shape[0]
        H_pad[i, :T_i] = t["h"].to(device)
        mask[i, :T_i] = True
    return H_pad, mask, T_list


def _whiten(H_pad, M_eigvecs, M_eigvals):
    """phi(x) = diag(sqrt(eigvals)) V^T x, applied to every token.
    Returns (n, T_max, r) if M given, else H_pad unchanged (r=d,
    isotropic metric)."""
    if M_eigvecs is None:
        return H_pad
    proj = H_pad @ M_eigvecs               # (n, T_max, r)
    return proj * M_eigvals.sqrt()          # (n, T_max, r)


def _metric_apply(H_pad, M_eigvecs, M_eigvals):
    """M @ h for every token, full d-dim (needed for the gradient's
    linear-decomposition trick -- NOT the same as _whiten, which only
    needs to preserve NORMS, not the actual d-dim vector)."""
    if M_eigvecs is None:
        return H_pad
    proj = H_pad @ M_eigvecs                # (n, T_max, r)
    scaled = proj * M_eigvals               # (n, T_max, r)
    return scaled @ M_eigvecs.T             # (n, T_max, d)


def _pairwise_kstate_chunk(phi_chunk, phi_flat, mask_chunk_flat, mask_flat,
                            sigma, kernel_type, laplace_eps=1e-3):
    """phi_chunk: (c*T_max, r) tokens for the current row-chunk.
    phi_flat: (N, r) ALL tokens (N = n*T_max). Returns masked k_state,
    shape (c*T_max, N) -- kernel value for every (row token, col token)
    pair in the chunk, zeroed at any padded position."""
    a_norm2 = (phi_chunk ** 2).sum(-1)              # (c*T_max,)
    b_norm2 = (phi_flat ** 2).sum(-1)               # (N,)
    cross = phi_chunk @ phi_flat.T                  # (c*T_max, N)
    dist2 = (a_norm2.unsqueeze(1) + b_norm2.unsqueeze(0) - 2 * cross).clamp(min=0)

    if kernel_type == "gaussian":
        k_state = torch.exp(-dist2 / (2 * sigma ** 2))
    elif kernel_type == "laplace":
        dist = torch.sqrt(dist2.clamp(min=laplace_eps ** 2))
        k_state = torch.exp(-dist / sigma)
    else:
        raise ValueError(f"unknown kernel_type: {kernel_type}")

    k_state = k_state * mask_chunk_flat.unsqueeze(1) * mask_flat.unsqueeze(0)
    return k_state


def bow_kernel_matrix_batched(trajectories, M_eigvecs, M_eigvals, sigma,
                               kernel_type="gaussian", chunk_size=32, device="cuda"):
    """Drop-in replacement for kernels.sequence_kernel_matrix(...,
    bag_of_words=True) -- same (n, n) output, computed via batched
    matmuls instead of an O(n^2) Python loop. ell_tau is not a
    parameter here since bag_of_words ignores position entirely."""
    n = len(trajectories)
    H_pad, mask, T_list = _pad_trajectories(trajectories, device)
    T_max = H_pad.shape[1]

    phi = _whiten(H_pad, M_eigvecs, M_eigvals)          # (n, T_max, r)
    phi_flat = phi.reshape(n * T_max, -1)
    mask_flat = mask.reshape(n * T_max)

    K = torch.zeros(n, n, device=device)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        c = end - start
        phi_chunk = phi[start:end].reshape(c * T_max, -1)
        mask_chunk_flat = mask[start:end].reshape(c * T_max)

        k_state = _pairwise_kstate_chunk(phi_chunk, phi_flat, mask_chunk_flat,
                                          mask_flat, sigma, kernel_type)
        summed = k_state.reshape(c, T_max, n, T_max).sum(dim=(1, 3))   # (c, n)

        T_i_chunk = T_list[start:end].float()
        T_j_all = T_list.float()
        K[start:end, :] = summed / (T_i_chunk.unsqueeze(1) * T_j_all.unsqueeze(0))

    return K.cpu()


def bow_gradients_batched(trajectories, alpha, M_eigvecs, M_eigvals, sigma,
                           kernel_type="gaussian", chunk_size=32, device="cuda"):
    """Drop-in replacement for sequence_rfm._compute_gradients(...,
    bag_of_words=True) -- same return type (list of n (T_i, d) tensors),
    computed via batched matmuls. `alpha` is the (n,) dual-coefficient
    tensor from the ridge-regression solve (same as the reference)."""
    n = len(trajectories)
    H_pad, mask, T_list = _pad_trajectories(trajectories, device)
    T_max = H_pad.shape[1]
    d = H_pad.shape[-1]

    phi = _whiten(H_pad, M_eigvecs, M_eigvals)
    phi_flat = phi.reshape(n * T_max, -1)
    mask_flat = mask.reshape(n * T_max)

    Mh = _metric_apply(H_pad, M_eigvecs, M_eigvals)     # (n, T_max, d)
    Mh_flat = (Mh * mask.unsqueeze(-1)).reshape(n * T_max, d)  # zero padded, defensive

    alpha_dev = alpha.to(device).float()
    T_list_f = T_list.float()
    # w_{j,s} = alpha_j / T_j, broadcast across all s (real positions only,
    # padded s already masked to 0 via mask_flat downstream)
    w_per_example = (alpha_dev / T_list_f)                      # (n,)
    w_flat = w_per_example.unsqueeze(1).expand(n, T_max).reshape(n * T_max)  # (N,)

    grads_flat = torch.zeros(n, T_max, d, device=device)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        c = end - start
        phi_chunk = phi[start:end].reshape(c * T_max, -1)
        mask_chunk_flat = mask[start:end].reshape(c * T_max)

        k_state = _pairwise_kstate_chunk(phi_chunk, phi_flat, mask_chunk_flat,
                                          mask_flat, sigma, kernel_type)      # (c*T_max, N)
        weighted = k_state * w_flat.unsqueeze(0)                             # (c*T_max, N)

        C_it = weighted.sum(dim=1)                                           # (c*T_max,)
        second_term = weighted @ Mh_flat                                     # (c*T_max, d)

        Mh_chunk = Mh[start:end].reshape(c * T_max, d)
        g_flat_chunk = Mh_chunk * C_it.unsqueeze(-1) - second_term           # (c*T_max, d)
        g_flat_chunk = g_flat_chunk * mask_chunk_flat.unsqueeze(-1)

        g_chunk = g_flat_chunk.reshape(c, T_max, d)
        T_i_chunk = T_list[start:end].float().unsqueeze(-1).unsqueeze(-1)    # (c,1,1)
        g_chunk = -g_chunk / (sigma ** 2 * T_i_chunk)

        grads_flat[start:end] = g_chunk

    # unpad back to the original list-of-(T_i, d) format. NOTE: kept on
    # `device`, NOT moved to .cpu() -- matches sequence_rfm._compute_gradients'
    # actual contract (it returns device-resident tensors; the caller's
    # loop right after this immediately does `a_it.to(device) * g`,
    # assuming g is already there). An earlier version of this function
    # incorrectly called .cpu() here, causing a "cuda:0 and cpu" device
    # mismatch crash one line into the caller -- verify_fast_bow.py's
    # CPU-only test didn't catch it since there was no device boundary
    # to cross in that test; see its updated version for a device check.
    grads = []
    for i in range(n):
        T_i = int(T_list[i].item())
        grads.append(grads_flat[i, :T_i])
    return grads