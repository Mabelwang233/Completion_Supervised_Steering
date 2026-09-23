"""
Sequence-RFM training loop (pdf Sections 4-5), single layer.

fit_sequence_rfm(trajectories, labels, ...) returns:
    directions: (d, k) tensor -- top-k steering subspace D_ell
    info: dict with diagnostics (per-iteration train accuracy, etc.)

No CLAS controller here by request -- steering uses a fixed coefficient
(alpha) swept over a grid in steer_and_evaluate.py, not a learned
contextual controller.
"""
import torch
from kernels import sequence_kernel_matrix, k_tau_rbf, state_kernel, apply_metric


def _metric_vec(diff, M_eigvecs, M_eigvals):
    """M @ diff for low-rank M = V diag(eigvals) V^T, returned as a
    full d-dim vector (not a scalar norm -- used inside the gradient,
    not the kernel value itself)."""
    if M_eigvecs is None:
        return diff
    proj = diff @ M_eigvecs                      # (..., k)
    scaled = proj * M_eigvals                    # (..., k)
    return scaled @ M_eigvecs.T                   # (..., d)


def _temporal_weights(tau, scheme="uniform"):
    if scheme == "uniform":
        return torch.ones_like(tau) / tau.shape[0]
    elif scheme == "linear_late":
        w = tau.clone()
        return w / w.sum().clamp(min=1e-8)
    else:
        raise ValueError(f"unknown temporal_weight scheme: {scheme}")


def _compute_gradients(trajectories, alpha, M_eigvecs, M_eigvals, sigma, ell_tau,
                        kernel_type="gaussian", bag_of_words=False):
    """
    g_i,t for every anchor position of every example (pdf Section 5):

    g_i,t = -(1/sigma^2 T_i) * sum_j (alpha_j / T_j) *
                sum_s k_tau(tau_i,t, tau_j,s) * k_M(h_i,t, h_j,s) * M(h_i,t - h_j,s)

    bag_of_words=True: drops the k_tau factor entirely (k_tau === 1),
    matching sequence_kernel's bag_of_words mode exactly.

    Returns a list (len n) of (T_i, d) gradient tensors.
    """
    n = len(trajectories)
    grads = []
    for i in range(n):
        h_i, tau_i = trajectories[i]["h"], trajectories[i]["tau"]
        T_i = h_i.shape[0]
        acc = torch.zeros_like(h_i)
        for j in range(n):
            h_j, tau_j = trajectories[j]["h"], trajectories[j]["tau"]
            T_j = h_j.shape[0]

            k_state = state_kernel(h_i, h_j, M_eigvecs, M_eigvals, sigma, kernel_type)  # (T_i, T_j)

            if bag_of_words:
                weight = k_state * (alpha[j].item() / T_j)
            else:
                w_tau = k_tau_rbf(tau_i, tau_j, ell_tau)              # (T_i, T_j)
                if w_tau.abs().sum() == 0:
                    continue
                weight = w_tau * k_state * (alpha[j].item() / T_j)    # (T_i, T_j)

            diff = h_i.unsqueeze(1) - h_j.unsqueeze(0)             # (T_i, T_j, d)
            m_diff = _metric_vec(diff, M_eigvecs, M_eigvals)       # (T_i, T_j, d)

            acc += (weight.unsqueeze(-1) * m_diff).sum(dim=1)      # (T_i, d)

        grads.append(-acc / (sigma ** 2 * T_i))
    return grads


def fit_sequence_rfm(trajectories, labels, sigma=1.0, lam=1e-3, k=1,
                      rfm_iters=3, ell_tau=0.15, temporal_weight="uniform",
                      kernel_type="gaussian", device="cpu", verbose=True,
                      bag_of_words=False, truncate_at_end=False,
                      fast_bow=False, fast_bow_chunk_size=32,
                      return_full_spectrum=False):
    """
    trajectories: list of n dicts {'h': (T_i, d) tensor, 'tau': (T_i,) tensor in [0,1]}
    labels: (n,) tensor, values in {-1, +1}

    bag_of_words=True: k_tau === 1 exactly (position ignored, "bag of
    prefix states" from pdf Section 3) -- a diagnostic baseline showing
    what sequence-RFM gets from pooling the trajectory alone, before any
    positional-alignment structure is added on top.

    fast_bow: batched/vectorized kernel+gradient computation instead of
    the O(n^2) Python loop (kernels.sequence_kernel_matrix /
    _compute_gradients above) -- dispatches to kernels_batched_bow.py
    when bag_of_words=True, or kernels_batched_positional.py (the
    k_tau-aware generalization) when bag_of_words=False. Same math
    either way, computed via a handful of large matmuls instead of
    thousands of tiny sequential GPU ops. See kernels_batched_bow.py's
    and kernels_batched_positional.py's module docstrings for the
    derivations. Run verify_fast_bow.py / verify_fast_bow_positional.py
    once on your end before trusting either path for a real run.

    truncate_at_end: controls WHEN the AGOP metric M gets truncated to
    rank-k during the rfm_iters loop -- these are two genuinely
    different estimators, not a speed/quality knob on the same one.

      False (default -- matches every sequence_rfm_directions.pt
      generated before this option existed): M is truncated to rank-k
      at the END OF EVERY iteration, and that truncated M is what feeds
      the KERNEL used in the next iteration. This means fits at
      different k DIVERGE starting at iteration 2 (different metric ->
      different kernel -> different alpha/gradients/Q), so a k=1 run
      and a k=5 run are NOT slices of one computation -- each k needs
      its own full fit_sequence_rfm call.

      True: M is carried at FULL economy rank (all min(d, n)
      components from the SVD of Q -- already computed in full by
      torch.linalg.svd every iteration regardless of k, so the SVD step
      itself costs nothing extra) through every iteration's KERNEL, and
      only truncated to the top-k requested directions ONCE, after the
      final iteration -- literally matching the pdf's Section 5 wording
      ("after iterating the usual RFM procedure, extract a steering
      subspace"). Under this mode every iteration's kernel/alpha/
      gradients/Q are IDENTICAL regardless of k, so ONE call at your
      largest desired k (e.g. k=5) produces a result whose top-1/3/5
      columns are exactly what separate k=1/3/5 calls would each have
      produced -- controller.py's load_directions_for_k already
      supports slicing a larger stored k down to a smaller one, so a
      single truncate_at_end=True, k=5 fit can serve k=1/3/5 downstream
      with no retraining.

      COST WARNING: the SVD is free either way, but the FULL-rank M
      also gets used inside state_kernel/apply_metric for the O(n^2*T^2)
      kernel evaluations in the NEXT iteration (kernels.py's own comment
      calls this "the expensive step") -- at full rank (up to n, e.g.
      ~200 training examples) instead of rank-k (e.g. 1 or 5), that
      per-pair metric computation is O(d*n) instead of O(d*k): roughly
      (n/k)x more expensive for every intermediate iteration's kernel +
      gradient computation. This is NOT free -- it trades a real
      compute cost for getting one fit to serve multiple k values (and
      for matching the pdf's literal iterate-then-truncate wording).

    return_full_spectrum: if True, stash the FULL economy eigendecomposition
      of M at the final iteration -- info["full_spectrum"] = {"eigenvalues":
      (r,), "eigenvectors": (d, r)} with r = min(d, n) -- before it gets
      truncated to the requested top-k below. This costs NOTHING extra: the
      SVD (U, S, _ = torch.linalg.svd(Q, ...)) is already computed at full
      economy rank every iteration regardless of k or truncate_at_end; this
      flag only controls whether we keep a copy of it at the last iteration
      instead of discarding everything past column k. Independent of the
      truncate_at_end cost tradeoff above (that only affects INTERMEDIATE
      iterations' kernel cost). Default False: zero change to existing
      behavior/output.
    """
    n = len(trajectories)
    y = labels.float().to(device)
    for t in trajectories:
        t["h"] = t["h"].to(device)
        t["tau"] = t["tau"].to(device)

    M_eigvecs, M_eigvals = None, None
    info = {"iters": [], "bag_of_words": bag_of_words, "truncate_at_end": truncate_at_end,
            "fast_bow": fast_bow}

    for it in range(rfm_iters):
        is_last_iter = (it == rfm_iters - 1)

        # 1. kernel matrix + ridge regression
        if fast_bow and bag_of_words:
            from kernels_batched_bow import bow_kernel_matrix_batched
            K = bow_kernel_matrix_batched(trajectories, M_eigvecs, M_eigvals, sigma,
                                           kernel_type=kernel_type, chunk_size=fast_bow_chunk_size,
                                           device=device).to(device)
        elif fast_bow and not bag_of_words:
            from kernels_batched_positional import positional_kernel_matrix_batched
            K = positional_kernel_matrix_batched(trajectories, M_eigvecs, M_eigvals, sigma, ell_tau,
                                                  kernel_type=kernel_type, chunk_size=fast_bow_chunk_size,
                                                  device=device).to(device)
        else:
            K = sequence_kernel_matrix(trajectories, M_eigvecs, M_eigvals, sigma, ell_tau,
                                        kernel_type, bag_of_words=bag_of_words).to(device)
        alpha = torch.linalg.solve(K + lam * torch.eye(n, device=device), y)

        train_pred = K @ alpha
        train_acc = ((train_pred > 0).float() == (y > 0).float()).float().mean().item()
        if verbose:
            tag = "bag-of-words" if bag_of_words else "sequence-RFM"
            if fast_bow:
                tag += " [fast]"
            rank_note = ""
            if truncate_at_end:
                rank_note = " [full-rank metric]" if M_eigvecs is not None and not is_last_iter else ""
            print(f"  [{tag}] iter {it+1}/{rfm_iters}  train_acc={train_acc:.3f}{rank_note}")
        info["iters"].append({"iter": it + 1, "train_acc": train_acc})

        # 2. gradients -> per-example aggregated direction q_i
        if fast_bow and bag_of_words:
            from kernels_batched_bow import bow_gradients_batched
            grads = bow_gradients_batched(trajectories, alpha, M_eigvecs, M_eigvals, sigma,
                                           kernel_type=kernel_type, chunk_size=fast_bow_chunk_size,
                                           device=device)
        elif fast_bow and not bag_of_words:
            from kernels_batched_positional import positional_gradients_batched
            grads = positional_gradients_batched(trajectories, alpha, M_eigvecs, M_eigvals, sigma, ell_tau,
                                                  kernel_type=kernel_type, chunk_size=fast_bow_chunk_size,
                                                  device=device)
        else:
            grads = _compute_gradients(trajectories, alpha, M_eigvecs, M_eigvals, sigma, ell_tau,
                                        kernel_type, bag_of_words=bag_of_words)
        q_list = []
        for i, g in enumerate(grads):
            a_it = _temporal_weights(trajectories[i]["tau"], temporal_weight).to(device)
            q_i = (a_it.unsqueeze(-1) * g).sum(dim=0)   # (d,)
            q_list.append(q_i)
        Q = torch.stack(q_list, dim=1)  # (d, n)

        # 3. AGOP update: M_new = (1/n) Q Q^T, via SVD of Q (avoids forming d x d).
        # This SVD is ALREADY the full economy decomposition (up to
        # min(d, n) components) regardless of k or truncate_at_end -- the
        # choice below is only about how much of it we KEEP for the next
        # iteration's kernel, not about the SVD's own cost.
        U, S, _ = torch.linalg.svd(Q, full_matrices=False)

        if return_full_spectrum and is_last_iter:
            # Full economy spectrum of M = (1/n) Q Q^T at the final
            # iteration, BEFORE the top-k truncation below. Already
            # computed above regardless of k/truncate_at_end -- this
            # just keeps a copy instead of letting it get discarded.
            info["full_spectrum"] = {
                "eigenvalues": ((S ** 2) / n).detach().cpu(),
                "eigenvectors": U.detach().cpu(),
            }

        if truncate_at_end and not is_last_iter:
            # Carry the FULL-RANK metric forward into the next
            # iteration's kernel (see cost warning in the docstring).
            M_eigvecs = U
            M_eigvals = (S ** 2) / n
        else:
            # truncate_at_end=False (original: truncate every iteration
            # so it feeds a rank-k kernel into the next round), OR this
            # IS the final iteration under truncate_at_end=True (truncate
            # here since this is what gets returned/exported either way).
            top_k = min(k, U.shape[1])
            M_eigvecs = U[:, :top_k]
            M_eigvals = (S[:top_k] ** 2) / n

    directions = M_eigvecs  # (d, k), final AGOP top-k eigenvectors
    return directions, info