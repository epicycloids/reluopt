"""Sketched pricing search for the practical RCG solver.

Pricing rho(nu) = max_{||u||<=1} nu^T (Xu)_+ only needs the SEARCH to be cheap;
each candidate is subsequently evaluated by a full-row numerical engine pass.
Neither the sketched search nor that rescore is a global pricing oracle.  Design:

  1. SEEDS: residual correlation X^T nu; top eigenvectors of X^T diag(nu)^2 X
     (shared by both signs); the strongest ACTIVE atoms + perturbations (local
     column generation); best violators of previous rounds; random directions
     in the top eigen-subspace; isotropic random.
  2. JOINT +-nu SEARCH: both signs ascend in ONE batch (per-column sign flag),
     so each sweep is one X-product for the union of candidates.
  3. SURVIVOR PRUNING: after a burn-in the batch is cut to the per-sign leaders.
  4. SKETCH: ascent runs on an importance row sketch (p_i ∝ |nu_i| ||x_i||,
     mixture-smoothed), either on the host or optionally on one Torch-engine
     device.  The device path transfers the sketch once and reuses it through
     polishing; host execution remains the default.
  5. ESCALATION: factored Burer-Monteiro SDP rounding (Goemans-Williamson) on
     the sketch when the cheap search finds no violator (hard instances).
  6. FULL-DATA RESCORE + POLISH: candidates are scored numerically on all rows
     (engine, float64 reduction); the top few get full-data ascent steps
     when the data is single-shard-cheap, else sketch-polished + rescored.
"""

from __future__ import annotations

import numpy as np

from .kernels import as_float32, importance_sample


# ---------------------------------------------------------------------- #
def _select_distinct(vfin, U, k, dedup_cos, rank_values=None, return_indices=False):
    """Keep the highest-scoring *oriented* directions.

    ReLU is not odd: ``relu(X @ u)`` and ``relu(X @ -u)`` are generally
    different atoms.  Only a positive near-unit cosine denotes a duplicate.
    """
    chosen = []
    ranking = np.asarray(vfin if rank_values is None else rank_values)
    for i in np.argsort(-ranking):
        u = np.asarray(U[:, i], dtype=np.float64)
        if any(float(u @ item[1]) > dedup_cos for item in chosen):
            continue
        chosen.append((float(vfin[i]), u.copy(), int(i)))
        if len(chosen) >= k:
            break
    if return_indices:
        return chosen
    return [(value, u) for value, u, _ in chosen]


def _predicted_decrease(correlation, sqnorm, lam):
    """Best one-coordinate objective decrease from a zero new coefficient."""
    correlation = np.asarray(correlation, dtype=np.float64)
    sqnorm = np.asarray(sqnorm, dtype=np.float64)
    return np.maximum(np.abs(correlation) - lam, 0.0) ** 2 / (
        2.0 * np.maximum(sqnorm, 1e-300)
    )


def _ascent_pm(
    Xs, nus, U0, sgn, n_iters=40, lr=0.3, prune_at=8, keep_min=24, trial_mode="both"
):
    """Joint batched ascent for both signs (host numpy).  Column j maximizes
    sgn_j * nus^T relu(Xs u_j).

    ``trial_mode='both'`` evaluates the fixed-region
    direction and a damped step separately.  ``'damped'`` and ``'fixed'``
    evaluate one trial and therefore use one forward product instead of two.
    ``'packed'`` makes the same choice while evaluating both trials in
    one wider product. ``'reuse'`` derives the damped projection from the
    current projection and the fixed-region trial, saving a forward product.
    ``'reuse-active'`` additionally omits stationary candidates on the host.
    All modes retain their best-so-far directions; the
    caller still performs full-data numerical rescoring.
    """
    if trial_mode == "reuse-active":
        return _ascent_pm_active(Xs, nus, U0, sgn, n_iters, lr, prune_at, keep_min)
    if trial_mode not in {"both", "damped", "fixed", "packed", "reuse"}:
        raise ValueError(f"unknown ascent trial_mode {trial_mode!r}")
    U = U0 / (np.linalg.norm(U0, axis=0, keepdims=True) + 1e-30)
    sg = np.asarray(sgn, dtype=Xs.dtype)

    def vals_mask(Ucur):
        Z = Xs @ Ucur
        Mk = Z >= 0
        return ((Z * Mk).T @ nus) * sg, Mk

    if trial_mode == "reuse":
        Z = Xs @ U
        mask = Z >= 0
        vals = (np.maximum(Z, 0).T @ nus) * sg
    else:
        vals, mask = vals_mask(U)
    best_vals = vals.copy()
    best_U = U.copy()
    for t in range(n_iters):
        G = (Xs.T @ (mask * nus[:, None])) * sg[None, :]
        gn = np.linalg.norm(G, axis=0, keepdims=True) + 1e-30
        br = G / gn
        gd = U + lr * br
        gd_norm = np.linalg.norm(gd, axis=0, keepdims=True) + 1e-30
        gd /= gd_norm
        if trial_mode == "reuse":
            # X(U + lr*br)/||U + lr*br|| needs no second product. Keep
            # projections aligned with their directions through trial choice
            # and survivor pruning. The original modes remain available for
            # trajectory comparisons near floating-point sign boundaries.
            zbr = Xs @ br
            zgd = (Z + lr * zbr) / gd_norm
            vbr = (np.maximum(zbr, 0).T @ nus) * sg
            vgd = (np.maximum(zgd, 0).T @ nus) * sg
            take = vbr >= vgd
            U = np.where(take[None, :], br, gd)
            Z = np.where(take[None, :], zbr, zgd)
            vals = np.where(take, vbr, vgd)
            mask = Z >= 0
        elif trial_mode == "damped":
            U = gd
            vals, mask = vals_mask(U)
        elif trial_mode == "fixed":
            U = br
            vals, mask = vals_mask(U)
        else:
            if trial_mode == "packed":
                q = br.shape[1]
                Z = Xs @ np.concatenate([br, gd], axis=1)
                M = Z >= 0
                vv = (Z * M).T @ nus
                vbr, vgd = vv[:q] * sg, vv[q:] * sg
                mbr, mgd = M[:, :q], M[:, q:]
            else:
                vbr, mbr = vals_mask(br)
                vgd, mgd = vals_mask(gd)
            take = vbr >= vgd
            U = np.where(take[None, :], br, gd)
            vals = np.where(take, vbr, vgd)
            mask = np.where(take[None, :], mbr, mgd)
        imp = vals > best_vals
        best_vals = np.where(imp, vals, best_vals)
        best_U = np.where(imp[None, :], U, best_U)
        if prune_at and t == prune_at and U.shape[1] > 2 * keep_min:
            keep = []
            for sv in (1.0, -1.0):
                cols = np.where(sg == sv)[0]
                if len(cols):
                    keep.extend(cols[np.argsort(-best_vals[cols])[:keep_min]].tolist())
            ki = np.asarray(sorted(keep))
            U, mask, sg = U[:, ki], mask[:, ki], sg[ki]
            if trial_mode == "reuse":
                Z = Z[:, ki]
            vals, best_vals, best_U = vals[ki], best_vals[ki], best_U[:, ki]
    return best_vals, best_U, sg


def _ascent_pm_active(Xs, nus, U0, sgn, n_iters, lr, prune_at, keep_min):
    """Projection reuse with CPU compaction of region-stationary candidates.

    Once a fixed-region maximizer stays inside its current region, its gradient
    and direction repeat forever. Keep that candidate's best result but remove
    it from subsequent products. Survivor pruning still ranks all candidates,
    including those already stationary. This is a host-only path because
    per-step dynamic compaction would synchronize CUDA execution.
    """
    U = U0 / (np.linalg.norm(U0, axis=0, keepdims=True) + 1e-30)
    signs = np.asarray(sgn, dtype=Xs.dtype)
    Z = Xs @ U
    best_vals = (np.maximum(Z, 0).T @ nus) * signs
    best_U = U.copy()
    active = np.arange(U.shape[1])
    for iteration in range(n_iters):
        if active.size:
            old_mask = Z >= 0
            sg = signs[active]
            G = (Xs.T @ (old_mask * nus[:, None])) * sg[None, :]
            br = G / (np.linalg.norm(G, axis=0, keepdims=True) + 1e-30)
            gd = U + lr * br
            gd_norm = np.linalg.norm(gd, axis=0, keepdims=True) + 1e-30
            gd /= gd_norm
            zbr = Xs @ br
            zgd = (Z + lr * zbr) / gd_norm
            vbr = (np.maximum(zbr, 0).T @ nus) * sg
            vgd = (np.maximum(zgd, 0).T @ nus) * sg
            take = vbr >= vgd
            U = np.where(take[None, :], br, gd)
            Z = np.where(take[None, :], zbr, zgd)
            vals = np.where(take, vbr, vgd)
            improved = vals > best_vals[active]
            best_vals[active[improved]] = vals[improved]
            best_U[:, active[improved]] = U[:, improved]
            stationary = take & np.all((zbr >= 0) == old_mask, axis=0)
            moving = ~stationary
            active, U, Z = active[moving], U[:, moving], Z[:, moving]
        if prune_at and iteration == prune_at and len(signs) > 2 * keep_min:
            keep = []
            for sign in (1.0, -1.0):
                cols = np.flatnonzero(signs == sign)
                keep.extend(cols[np.argsort(-best_vals[cols])[:keep_min]])
            keep = np.asarray(sorted(keep))
            inverse = np.full(len(signs), -1, dtype=np.intp)
            inverse[keep] = np.arange(len(keep))
            surviving = inverse[active] >= 0
            active = inverse[active[surviving]]
            U, Z = U[:, surviving], Z[:, surviving]
            signs, best_vals, best_U = signs[keep], best_vals[keep], best_U[:, keep]
        if not active.size and (not prune_at or iteration >= prune_at or prune_at >= n_iters):
            break
    return best_vals, best_U, signs


def _ascent_pm_torch(
    Xs,
    nus,
    U0,
    sgn,
    n_iters=40,
    lr=0.3,
    prune_at=8,
    keep_min=24,
    trial_mode="both",
    device=None,
    _return_tensors=False,
    _sign_cols=None,
):
    """Torch equivalent of :func:`_ascent_pm` on one selected device.

    ``Xs`` and ``nus`` may already be device tensors; in that case they are
    reused without a host round trip.  All iteration state remains on
    ``device`` and only the final best values, directions, and signs are
    returned to NumPy. Trial modes mirror the host implementation; dynamic
    ``reuse-active`` compaction is only available on the host.
    """
    if trial_mode not in {"both", "damped", "fixed", "packed", "reuse"}:
        raise ValueError(f"unknown ascent trial_mode {trial_mode!r}")
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised without Torch
        raise RuntimeError("device pricing search requires Torch") from exc

    if device is None:
        device = Xs.device if torch.is_tensor(Xs) else "cpu"
    dev = torch.device(device)
    if torch.is_tensor(Xs):
        Xd = Xs.to(device=dev, dtype=torch.float32)
    else:
        Xd = torch.as_tensor(np.asarray(Xs), dtype=torch.float32, device=dev)
    if torch.is_tensor(nus):
        nd = nus.to(device=dev, dtype=Xd.dtype)
    else:
        nd = torch.as_tensor(np.asarray(nus), dtype=Xd.dtype, device=dev)
    if torch.is_tensor(U0):
        U = U0.to(device=dev, dtype=Xd.dtype)
    else:
        U = torch.as_tensor(np.asarray(U0), dtype=Xd.dtype, device=dev)
    if torch.is_tensor(sgn):
        sg = sgn.to(device=dev, dtype=Xd.dtype)
    else:
        sg = torch.as_tensor(np.asarray(sgn), dtype=Xd.dtype, device=dev)

    with torch.no_grad():
        U = U / (torch.linalg.vector_norm(U, dim=0, keepdim=True) + 1e-30)

        def vals_mask(Ucur):
            Z = Xd @ Ucur
            Mk = Z >= 0
            return ((Z * Mk).T @ nd) * sg, Mk

        if trial_mode == "reuse":
            Z = Xd @ U
            mask = Z >= 0
            vals = (torch.clamp_min(Z, 0).T @ nd) * sg
        else:
            vals, mask = vals_mask(U)
        best_vals = vals.clone()
        best_U = U.clone()
        for t in range(n_iters):
            G = (Xd.T @ (mask * nd[:, None])) * sg[None, :]
            gn = torch.linalg.vector_norm(G, dim=0, keepdim=True) + 1e-30
            br = G / gn
            gd = U + lr * br
            gd_norm = torch.linalg.vector_norm(gd, dim=0, keepdim=True) + 1e-30
            gd = gd / gd_norm
            if trial_mode == "reuse":
                zbr = Xd @ br
                zgd = (Z + lr * zbr) / gd_norm
                vbr = (torch.clamp_min(zbr, 0).T @ nd) * sg
                vgd = (torch.clamp_min(zgd, 0).T @ nd) * sg
                take = vbr >= vgd
                U = torch.where(take[None, :], br, gd)
                Z = torch.where(take[None, :], zbr, zgd)
                vals = torch.where(take, vbr, vgd)
                mask = Z >= 0
            elif trial_mode == "damped":
                U = gd
                vals, mask = vals_mask(U)
            elif trial_mode == "fixed":
                U = br
                vals, mask = vals_mask(U)
            else:
                if trial_mode == "packed":
                    q = br.shape[1]
                    Z = Xd @ torch.cat([br, gd], dim=1)
                    M = Z >= 0
                    vv = (Z * M).T @ nd
                    vbr, vgd = vv[:q] * sg, vv[q:] * sg
                    mbr, mgd = M[:, :q], M[:, q:]
                else:
                    vbr, mbr = vals_mask(br)
                    vgd, mgd = vals_mask(gd)
                take = vbr >= vgd
                U = torch.where(take[None, :], br, gd)
                vals = torch.where(take, vbr, vgd)
                mask = torch.where(take[None, :], mbr, mgd)
            imp = vals > best_vals
            best_vals = torch.where(imp, vals, best_vals)
            best_U = torch.where(imp[None, :], U, best_U)
            if prune_at and t == prune_at and U.shape[1] > 2 * keep_min:
                keep = []
                for sign_index, sv in enumerate((1.0, -1.0)):
                    cols = (
                        torch.nonzero(sg == sv, as_tuple=False).flatten()
                        if _sign_cols is None
                        else _sign_cols[sign_index]
                    )
                    if cols.numel():
                        order = torch.argsort(best_vals[cols], descending=True)
                        keep.append(cols[order[:keep_min]])
                ki = torch.sort(torch.cat(keep))[0]
                U, mask, sg = U[:, ki], mask[:, ki], sg[ki]
                if trial_mode == "reuse":
                    Z = Z[:, ki]
                vals = vals[ki]
                best_vals, best_U = best_vals[ki], best_U[:, ki]

    if _return_tensors:
        return best_vals, best_U, sg
    return (
        best_vals.detach().cpu().numpy(),
        best_U.detach().cpu().numpy(),
        sg.detach().cpu().numpy(),
    )


def _factored_sdp_round_host(Xs, absnu, rng, r=16, n_power=10, n_hyper=48):
    """Burer-Monteiro Max-Cut SDP + GW rounding on the (host) sketch rows."""
    n, _d = Xs.shape
    B = absnu[:, None] * Xs
    L = rng.standard_normal((r, n))
    L /= np.linalg.norm(L, axis=0, keepdims=True) + 1e-12
    for _ in range(n_power):
        grad = (L @ B) @ B.T
        L = grad / (np.linalg.norm(grad, axis=0, keepdims=True) + 1e-12)
    E = np.sign(L.T @ rng.standard_normal((r, n_hyper)))
    E[E == 0] = 1.0
    D = B.T @ E
    return D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-30)


def spectral_seeds(engine, nu_sh, k=4):
    """Top-k eigvecs of X^T diag(nu)^2 X via one engine reduction."""
    S = engine.spectral(nu_sh)
    _w, Q = np.linalg.eigh(0.5 * (S + S.T))
    d = S.shape[0]
    cols = []
    for j in range(1, min(k, d) + 1):
        cols.append(Q[:, -j])
        cols.append(-Q[:, -j])
    return np.stack(cols, axis=1), Q[:, -min(k, d) :]


def _sketch_spectral(Xs, weights, k=4, method="exact"):
    """Leading weighted sketch directions, used only as search seeds.

    ``randomized`` applies six block-power iterations to the weighted design
    instead of forming a d-by-d Gram matrix. An oversampled subspace and final
    Rayleigh--Ritz rotation improve the leading directions. Its independent
    fixed seed leaves the caller's random search stream unchanged.
    """
    if method not in {"exact", "randomized"}:
        raise ValueError(f"unknown spectral_method {method!r}")
    X64 = np.asarray(Xs, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    d = X64.shape[1]
    k = min(k, d)
    if method == "exact" or d <= 2 * k:
        S = X64.T @ (weight[:, None] * X64)
        _, Q = np.linalg.eigh(0.5 * (S + S.T))
        Q = Q[:, -k:]
    else:
        width = min(d, max(2 * k, k + 8))
        Q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((d, width)))
        for _ in range(6):
            Z = X64 @ Q
            Q, _ = np.linalg.qr(X64.T @ (weight[:, None] * Z))
        Z = X64 @ Q
        S = Z.T @ (weight[:, None] * Z)
        _, V = np.linalg.eigh(0.5 * (S + S.T))
        Q = Q @ V[:, -k:]
    cols = np.empty((d, 2 * k), dtype=np.float64)
    cols[:, 0::2] = Q[:, ::-1]
    cols[:, 1::2] = -Q[:, ::-1]
    return cols, Q


def _build_seeds(
    c0,
    rng,
    U_active,
    beta_active,
    prev_dirs,
    spec_cols,
    Qtop,
    d,
    n_random,
    n_subspace,
    n_perturb,
    max_active_seeds,
):
    seeds = []
    nc = np.linalg.norm(c0)
    if nc > 0:
        seeds.append((c0 / nc)[:, None])
    seeds.append(spec_cols)
    if U_active is not None and U_active.shape[1] > 0:
        Ua = np.asarray(U_active, float)
        if Ua.shape[1] > max_active_seeds:
            if beta_active is not None:
                pick = np.argsort(-np.abs(np.asarray(beta_active)))[:max_active_seeds]
            else:
                pick = np.arange(max_active_seeds)
            Ua = Ua[:, pick]
        seeds.append(Ua)
        for t in range(n_perturb):
            seeds.append(Ua + 0.07 * (2.0**t) * rng.standard_normal(Ua.shape))
    if prev_dirs:
        seeds.append(np.stack(prev_dirs, axis=1))
    if n_subspace > 0 and Qtop is not None and Qtop.shape[1] >= 2:
        seeds.append(Qtop @ rng.standard_normal((Qtop.shape[1], n_subspace)))
    if n_random > 0:
        seeds.append(rng.standard_normal((d, n_random)))
    U0 = np.concatenate([s.reshape(d, -1) for s in seeds], axis=1)
    return U0 / (np.linalg.norm(U0, axis=0, keepdims=True) + 1e-30)


def search_signed_atoms(
    engine,
    nu_sh,
    nu_host,
    lam,
    k=8,
    rng=None,
    U_active=None,
    beta_active=None,
    prev_dirs=None,
    spec=None,
    n_random=16,
    n_subspace=8,
    n_perturb=1,
    max_active_seeds=12,
    sketch_m=32768,
    iters_sketch=40,
    iters_full=8,
    top_polish=6,
    lr=0.3,
    use_sdp=False,
    sdp_r=16,
    sdp_power=10,
    sdp_hyper=48,
    dedup_cos=0.999,
    row_norms=None,
    ascent_trial="auto",
    candidate_order="correlation",
    search_backend="host",
    retain_finalist_features=True,
    retained_feature_max_bytes=256 * 1024 * 1024,
    spectral_method="exact",
):
    """Search both +nu and -nu, then numerically rescore on all rows.

    ``search_backend='host'`` uses NumPy sketch ascent.
    ``'device'`` transfers the sketch once to the first device of a Torch
    engine and reuses those tensors for polishing; ``'auto'`` selects that
    path for Torch engines and the host path otherwise.  Search location never
    changes which full-row rescore is performed or its numerical semantics.
    ``'device-graph'`` uses an engine-local two-entry CUDA graph cache; it has
    an initial capture cost and requires a CUDA device. ``spectral_method``
    selects exact Gram eigenvectors or randomized block-power search seeds.
    ``ascent_trial='auto'`` uses projection reuse, with host stationary-candidate
    compaction for sketches with at least 8192 rows and at most 16 dimensions.
    ``'both'`` retains the original two-trial calculation for comparisons.

    ``retain_finalist_features=True`` keeps the full-data feature columns
    formed by the final rescore, reduces
    them immediately to the ranked distinct additions, and returns an aligned
    cache for the restricted master.  A conservative payload estimate must fit
    ``retained_feature_max_bytes`` or the ordinary rebuild path is used.
    """
    if candidate_order not in {"correlation", "decrease"}:
        raise ValueError(f"unknown candidate_order {candidate_order!r}")
    if search_backend not in {"host", "device", "device-graph", "auto"}:
        raise ValueError(f"unknown search_backend {search_backend!r}")
    if spectral_method not in {"exact", "randomized"}:
        raise ValueError(f"unknown spectral_method {spectral_method!r}")
    retained_feature_max_bytes = int(retained_feature_max_bytes)
    if retained_feature_max_bytes < 0:
        raise ValueError("retained_feature_max_bytes must be nonnegative")
    resolved_search = search_backend
    if resolved_search == "auto":
        resolved_search = "device" if engine.backend == "torch" else "host"
    if resolved_search in {"device", "device-graph"} and engine.backend != "torch":
        raise ValueError("device pricing search requires a Torch engine")
    if ascent_trial not in {"auto", "both", "damped", "fixed", "packed", "reuse", "reuse-active"}:
        raise ValueError(f"unknown ascent_trial {ascent_trial!r}")
    resolved_trial = ascent_trial
    if resolved_trial == "reuse-active" and resolved_search != "host":
        raise ValueError("ascent_trial='reuse-active' requires host search")
    rng = rng or np.random.default_rng(0)
    n, d = engine.n, engine.d
    c0 = engine.xtv(nu_sh)

    # ---- host sketch ----
    m = sketch_m if (sketch_m and 2 * sketch_m < n) else n
    if resolved_trial == "auto":
        resolved_trial = (
            "reuse-active"
            if resolved_search == "host" and d <= 16 and m >= 8192
            else "reuse"
        )
    if m < n:
        if row_norms is None:
            row_norms = np.linalg.norm(np.asarray(engine._Xhost, dtype=np.float64), axis=1)
        idx, swt = importance_sample(nu_host, row_norms, m, rng)
        Xs, _ = engine.gather_rows(idx)
        Xs = as_float32(Xs)
        nus = as_float32(nu_host[idx] * swt)
        spec_w = (nu_host[idx] ** 2) * swt  # unbiased for nu_i^2 sums
    else:
        Xs = as_float32(engine._Xhost)
        nus = as_float32(nu_host)
        spec_w = nu_host**2

    if spec is None:
        # spectral seeds from the sketch (seeds only -- no validity at stake)
        spec = _sketch_spectral(Xs, spec_w, method=spectral_method)
    spec_cols, Qtop = spec

    Up = _build_seeds(
        c0,
        rng,
        U_active,
        beta_active,
        prev_dirs,
        spec_cols,
        Qtop,
        d,
        n_random,
        n_subspace,
        n_perturb,
        max_active_seeds,
    )
    Um = _build_seeds(
        -c0,
        rng,
        U_active,
        beta_active,
        None if not prev_dirs else [-u for u in prev_dirs],
        spec_cols,
        Qtop,
        d,
        n_random,
        n_subspace,
        n_perturb,
        max_active_seeds,
    )
    if use_sdp:
        Urd = _factored_sdp_round_host(
            Xs, np.abs(nus), rng, r=sdp_r, n_power=sdp_power, n_hyper=sdp_hyper
        )
        Up = np.concatenate([Up, Urd], axis=1)
        Um = np.concatenate([Um, Urd], axis=1)

    U0 = np.concatenate([Up, Um], axis=1).astype(np.float32)
    sgn = np.concatenate([np.ones(Up.shape[1]), -np.ones(Um.shape[1])])
    if resolved_search in {"device", "device-graph"}:
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - Torch engine implies import
            raise RuntimeError("device pricing search requires Torch") from exc
        search_device = engine.devices[0]
        Xs_search = torch.as_tensor(Xs, dtype=torch.float32, device=search_device)
        nus_search = torch.as_tensor(nus, dtype=torch.float32, device=search_device)
        if resolved_search == "device-graph":
            from ._cuda_pricing import ascent_cuda_graph

            if not hasattr(engine, "_pricing_graphs"):
                from collections import OrderedDict

                engine._pricing_graphs = OrderedDict()
            ascent = ascent_cuda_graph
            ascent_extra = {"device": search_device, "graph_cache": engine._pricing_graphs}
        else:
            ascent = _ascent_pm_torch
            ascent_extra = {"device": search_device}
    else:
        Xs_search, nus_search = Xs, nus
        ascent = _ascent_pm
        ascent_extra = {}
    _, Usearch, sg_s = ascent(
        Xs_search,
        nus_search,
        U0,
        sgn,
        n_iters=iters_sketch,
        lr=lr,
        trial_mode=resolved_trial,
        **ascent_extra,
    )

    # ---- full-row numerical rescore; sketch-polish per-sign top; rescore ---
    out = {}
    Us64 = Usearch.astype(np.float64)
    Us64 /= np.linalg.norm(Us64, axis=0, keepdims=True) + 1e-30
    vals = engine.rescore(Us64, nu_sh) * sg_s
    finalists = []
    for key, sv in (("plus", 1.0), ("minus", -1.0)):
        cols = np.where(sg_s == sv)[0]
        order = cols[np.argsort(-vals[cols])][: max(top_polish, k)]
        Utop = Us64[:, order]
        if iters_full > 0 and len(order):
            _, Upol, _ = ascent(
                Xs_search,
                nus_search,
                Utop.astype(np.float32),
                np.full(len(order), sv),
                n_iters=iters_full,
                lr=lr * 0.5,
                prune_at=0,
                trial_mode=resolved_trial,
                **ascent_extra,
            )
            Utop = np.concatenate([Utop, Upol.astype(np.float64)], axis=1)
            Utop /= np.linalg.norm(Utop, axis=0, keepdims=True) + 1e-30
        finalists.append((key, sv, Utop))

    # The cap covers retained feature payload, including conservative
    # transient copies while each full finalist block is reduced to k columns
    # and the two sign blocks are joined.  The actual allocator overhead is
    # backend-dependent and is intentionally reported separately by the
    # benchmark harness.
    col_bytes = engine.feature_buffer_bytes(1)
    prefix_selected = 0
    peak_cols = 0
    for _, _, Utop in finalists:
        selected = min(k, Utop.shape[1])
        peak_cols = max(peak_cols, prefix_selected + Utop.shape[1] + selected)
        prefix_selected += selected
    peak_cols = max(peak_cols, 2 * prefix_selected)
    required_peak_bytes = int(peak_cols * col_bytes)
    retain = bool(
        retain_finalist_features and required_peak_bytes <= retained_feature_max_bytes
    )

    additions = []
    selected_feature_blocks = []
    selected_aty = []
    selected_sqnorm = []
    selected_dual_corr = []
    for key, sv, Utop in finalists:
        stats = None
        if retain:
            stats = engine.rescore_finalists(
                Utop,
                nu_sh,
                retain_features=True,
                retained_feature_max_bytes=retained_feature_max_bytes,
            )
            corr, sqnorm = stats["corr"], stats["sqnorm"]
            vfin = corr * sv
            gain = (
                _predicted_decrease(corr, sqnorm, lam)
                if candidate_order == "decrease"
                else vfin
            )
        elif candidate_order == "decrease":
            corr, sqnorm = engine.rescore_stats(Utop, nu_sh)
            vfin = corr * sv
            gain = _predicted_decrease(corr, sqnorm, lam)
        else:
            vfin = engine.rescore(Utop, nu_sh) * sv
            gain = vfin
        out[key] = _select_distinct(vfin, Utop, k, dedup_cos)
        ranked = _select_distinct(
            vfin, Utop, k, dedup_cos, rank_values=gain, return_indices=True
        )
        ranked_indices = [idx for _, _, idx in ranked]
        for value, u, idx in ranked:
            score = float(gain[idx]) if candidate_order == "decrease" else value
            additions.append((score, value, u))
        if retain and ranked_indices:
            selected_feature_blocks.append(
                engine.slice_cols(stats["features"], ranked_indices)
            )
            selected_aty.append(np.asarray(stats["aty"])[ranked_indices])
            selected_sqnorm.append(np.asarray(stats["sqnorm"])[ranked_indices])
            selected_dual_corr.append(np.asarray(stats["corr"])[ranked_indices])
        stats = None  # release the full finalist block before the next sign

    addition_cache = None
    if retain and selected_feature_blocks:
        if len(selected_feature_blocks) == 1:
            A_add = selected_feature_blocks[0]
        elif engine.backend == "numpy":
            A_add = [
                np.concatenate(parts, axis=1) for parts in zip(*selected_feature_blocks)
            ]
        else:
            import torch  # type: ignore

            A_add = [torch.cat(parts, dim=1) for parts in zip(*selected_feature_blocks)]
        addition_cache = {
            "A_sh": A_add,
            "aty": np.concatenate(selected_aty),
            "sqnorm": np.concatenate(selected_sqnorm),
            "dual_corr": np.concatenate(selected_dual_corr),
        }
    out["addition"] = additions
    out["addition_cache"] = addition_cache
    out["feature_reuse"] = {
        "requested": bool(retain_finalist_features),
        "retained": bool(addition_cache is not None),
        "fallback_reason": (
            None
            if addition_cache is not None
            else "cap_exceeded"
            if retain_finalist_features and not retain
            else "no_additions"
            if retain_finalist_features
            else "disabled"
        ),
        "cap_bytes": retained_feature_max_bytes,
        "required_peak_payload_bytes": required_peak_bytes,
        "retained_payload_bytes": (
            len(additions) * col_bytes if addition_cache is not None else 0
        ),
        "retained_columns": len(additions) if addition_cache is not None else 0,
    }
    out["best"] = max(
        [v for v, _ in out["plus"]] + [v for v, _ in out["minus"]], default=0.0
    )
    return out
