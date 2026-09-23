"""
Kernel primitives for sequence-RFM (pdf Sections 3-5).

Design decisions, documented here so they're not silently buried in code:

1. State kernel is Gaussian (squared Mahalanobis distance / 2*sigma^2), NOT
   the Laplace kernel (sqrt distance / L) that the original single-vector
   RFM paper uses internally (kernel='l2_high_dim' in their xRFM package).
   We discussed this tradeoff at length: the Laplace kernel's gradient has
   a 1/D(a,b) singularity that blows up on near-duplicate adjacent-timestep
   activations, which are common within a single completion trajectory.
   The Gaussian form avoids that and is what the gradient formula in the
   pdf is actually consistent with. Treat 'gaussian' vs 'laplace' as a
   config switch if you want to ablate this later -- laplace is provided
   but not the default, and needs an eps floor on the distance to avoid
   div-by-zero (implemented below).

2. M is never formed as a dense d x d matrix. We only ever need
   (a) top-k eigenvectors of M for the final steering subspace, and
   (b) M applied inside the kernel as a metric for the NEXT rfm iteration.
   Both only require a low-rank factor (eigvecs: d x k, eigvals: k), so we
   carry M around as that factor throughout, never materializing d x d.
   Iteration 0 uses M = identity (plain Euclidean/isotropic kernel).

3. k_tau is RBF on normalized position tau_i = t / (T_i - 1), with sparse
   thresholding: pairs with weight below `tau_sparsity_eps` are zeroed,
   which is what keeps the O(n^2 * T^2) sum tractable once combined with
   anchor-position subsampling (see extract_trajectories.py).

4. bag_of_words mode: k_tau === 1 exactly (position ignored entirely),
   implemented as a genuine bypass -- NOT an approximation via a huge
   ell_tau. Setting ell_tau very large would still run it through the
   sparsity threshold and floating-point exp(), for no benefit; skipping
   k_tau_rbf entirely is both cheaper and exact. This is the baseline
   from pdf Section 4 ("if k_tau === 1, this treats the sequence as a bag
   of prefix states") -- useful as a diagnostic for whether k_tau's
   positional structure is actually contributing anything over plain
   mean-pooling of the trajectory.
"""
import torch


def k_tau_rbf(tau_i, tau_j, ell_tau, sparsity_eps=1e-4):
    """
    RBF kernel on normalized position.
    tau_i: (T_i,) tensor in [0, 1]
    tau_j: (T_j,) tensor in [0, 1]
    returns: (T_i, T_j) weight matrix, sparsified (near-zero entries -> 0)
    """
    diff = tau_i.unsqueeze(1) - tau_j.unsqueeze(0)  # (T_i, T_j)
    w = torch.exp(-(diff ** 2) / (2 * ell_tau ** 2))
    w = torch.where(w < sparsity_eps, torch.zeros_like(w), w)
    return w


def apply_metric(diff, M_eigvecs, M_eigvals):
    """
    Compute the metric-weighted squared norm ||diff||_M^2 = diff^T M diff
    using a low-rank M = V diag(eigvals) V^T, WITHOUT forming M explicitly.

    diff: (..., d)
    M_eigvecs: (d, k) or None (None => isotropic, M = identity)
    M_eigvals: (k,) or None
    returns: (...,) squared metric norm
    """
    if M_eigvecs is None:
        return (diff ** 2).sum(dim=-1)
    proj = diff @ M_eigvecs               # (..., k)
    return (M_eigvals * proj ** 2).sum(dim=-1)


def state_kernel(h_i, h_j, M_eigvecs, M_eigvals, sigma, kernel_type="gaussian",
                  laplace_eps=1e-3):
    """
    Per-pair state kernel k_M(a, b).
    h_i: (T_i, d), h_j: (T_j, d)
    returns: (T_i, T_j)
    """
    diff = h_i.unsqueeze(1) - h_j.unsqueeze(0)     # (T_i, T_j, d)
    sq_dist = apply_metric(diff, M_eigvecs, M_eigvals)  # (T_i, T_j)

    if kernel_type == "gaussian":
        return torch.exp(-sq_dist / (2 * sigma ** 2))
    elif kernel_type == "laplace":
        dist = torch.sqrt(torch.clamp(sq_dist, min=laplace_eps ** 2))
        return torch.exp(-dist / sigma)
    else:
        raise ValueError(f"unknown kernel_type: {kernel_type}")


def sequence_kernel(traj_i, traj_j, M_eigvecs, M_eigvals, sigma, ell_tau,
                     kernel_type="gaussian", bag_of_words=False):
    """
    K_M(P_i, P_j) -- pdf Section 3, kernel-mean / convolution-style kernel.
    traj_i, traj_j: dicts with keys 'h' (T, d) and 'tau' (T,)

    bag_of_words=True: k_tau === 1 exactly (skip position entirely) --
    the ell_tau -> infinity limit, but exact rather than approximated.
    returns: scalar
    """
    h_i, tau_i = traj_i["h"], traj_i["tau"]
    h_j, tau_j = traj_j["h"], traj_j["tau"]

    k_state = state_kernel(h_i, h_j, M_eigvecs, M_eigvals, sigma, kernel_type)

    if bag_of_words:
        return k_state.sum() / (h_i.shape[0] * h_j.shape[0])

    w_tau = k_tau_rbf(tau_i, tau_j, ell_tau)          # (T_i, T_j)
    if w_tau.abs().sum() == 0:
        return torch.tensor(0.0, device=h_i.device)

    return (w_tau * k_state).sum() / (h_i.shape[0] * h_j.shape[0])


def sequence_kernel_matrix(trajectories, M_eigvecs, M_eigvals, sigma, ell_tau,
                            kernel_type="gaussian", bag_of_words=False):
    """
    Full n x n kernel matrix. O(n^2) sequence-kernel evaluations, each
    O(T_i * T_j) internally -- this is the expensive step, see the
    project's timing notes for why anchor subsampling matters.
    """
    n = len(trajectories)
    K = torch.zeros(n, n)
    for i in range(n):
        for j in range(i, n):
            val = sequence_kernel(trajectories[i], trajectories[j],
                                   M_eigvecs, M_eigvals, sigma, ell_tau, kernel_type,
                                   bag_of_words=bag_of_words)
            K[i, j] = val
            K[j, i] = val
    return K
