"""Joint direction proposals with full-data acceptance owned by the caller.

This is a proposal mechanism for moving several active directions together.
It freezes each active atom's mask m_j = 1{X u_j >= 0} and approximately solves
the group-Lasso over the resulting chamber-restricted weight vectors,

    min_Z 1/2 || sum_j m_j ⊙ (X z_j) - y ||^2 + lam sum_j ||z_j||_2

(the gated-ReLU restricted master).  The numerical iterate is projected back to
valid ReLU directions (u_j = z_j/||z_j||).  The caller then performs a full-data
numerical finite-master refit.  Full working-set replacement requires a computed
objective no larger than the incumbent; after a losing replacement trial, the
caller may instead append a few proposal directions and refit.  This mechanism
does not turn the approximate master or the overall production loop into the
ideal exact algorithm.

The frozen-mask proposals use float32 host sketches. An optional float64
L-BFGS proposal directly refines the signed ReLU objective, allowing activation
masks to change. Neither local proposal supplies a global optimality bound.
"""

from __future__ import annotations

import numpy as np

from .kernels import as_float32


def gated_group_lasso(
    Xs, ys, M, lam, Z0=None, iters=120, tol=1e-7, row_w=None, check_every=0,
    rng_seed=0, method="coordinate",
):
    """Block coordinate descent on the frozen-mask group-Lasso (host).

    Xs: m x d (float32), M: m x P {0,1} masks.  Each block update is a few
    proximal-gradient steps with a block-specific Lipschitz estimate.  `iters`
    controls the approximate sweep budget.  Returns the numerical iterate and
    its frozen-mask objective value."""
    if method == "accelerated":
        return _accelerated_group_lasso(Xs, ys, M, lam, Z0, iters, tol)
    if method != "coordinate":
        raise ValueError(f"unknown refinement method {method!r}")
    m, d = Xs.shape
    P = M.shape[1]
    dt = Xs.dtype
    Z = np.zeros((d, P), dtype=dt) if Z0 is None else np.asarray(Z0, dtype=dt).copy()
    # A block's mask and coefficients are used repeatedly as column vectors.
    M = np.asfortranarray(M)
    # per-block d x d Grams and Lipschitz constants (one batched pass)
    Qs = []
    Ls = np.empty(P)
    for j in range(P):
        Q = Xs.T @ (M[:, j : j + 1] * Xs)
        Qs.append(Q)
        Ls[j] = max(float(np.linalg.eigvalsh(0.5 * (Q + Q.T))[-1]), 1e-12)
    r = _frozen_residual(Xs, Z, M, ys)

    def fobj(Zc, rc):
        return _group_objective(Zc, rc, lam)

    last = None
    sweeps = max(8, iters // 3)  # one CD sweep ~ 3 FISTA iterations of work
    if m > 100_000:  # large sketches are memory-bound: fewer,
        sweeps = min(sweeps, 12)  # cheaper sweeps with a looser tolerance
        tol = max(tol, 1e-5)
    for sw in range(sweeps):
        for j in range(P):
            old = Z[:, j].copy()
            zj = old
            # X_j.T @ (r - X_j @ z_j) = X_j.T @ r - Q_j @ z_j.
            # This removes the old full-row X @ z_j from every block update.
            cj = Xs.T @ (M[:, j] * r) - Qs[j] @ old
            for _inner in range(3):
                g = Qs[j] @ zj + cj
                znew = zj - g / Ls[j]
                nz = float(np.linalg.norm(znew))
                znew *= max(0.0, 1.0 - (lam / Ls[j]) / max(nz, 1e-30))
                if np.array_equal(znew, zj):
                    break
                zj = znew
            Z[:, j] = zj
            delta = zj - old
            if np.any(delta):
                r += (Xs @ delta) * M[:, j]
        # Refresh from the actual iterate before using its objective to stop.
        # This also prevents accumulation of float32 residual-update drift.
        r = _frozen_residual(Xs, Z, M, ys)
        obj = fobj(Z, r)
        if last is not None and abs(last - obj) <= tol * max(1.0, abs(obj)):
            break
        last = obj
    return Z, fobj(Z, r)


def _frozen_residual(X, Z, M, y):
    return np.einsum("ij,ij->i", X @ Z, M) - y


def _group_objective(Z, residual, lam):
    r64 = np.asarray(residual, dtype=np.float64)
    z64 = np.asarray(Z, dtype=np.float64)
    return 0.5 * float(r64 @ r64) + lam * float(np.linalg.norm(z64, axis=0).sum())


def _accelerated_group_lasso(X, y, M, lam, Z0, iters, tol):
    """Monotone accelerated proximal gradients using two matrix products.

    The Hessian bound ``max_i ||M[i]||² * ||X||²`` follows directly from
    Cauchy--Schwarz. Backtracking checks the quadratic majorizer and permits
    smaller local bounds; an objective increase restarts acceleration.
    Unlike block descent, this path never forms one Gram per activation mask.
    """
    d, p = X.shape[1], M.shape[1]
    Z = np.zeros((d, p), dtype=X.dtype) if Z0 is None else np.array(Z0, dtype=X.dtype, copy=True)
    r = _frozen_residual(X, Z, M, y)
    objective = _group_objective(Z, r, lam)
    if not p:
        return Z, objective
    gram = X.T @ X
    spectral = max(float(np.linalg.eigvalsh(0.5 * (gram + gram.T))[-1]), 0.0)
    mask_norm = float(np.max(np.einsum("ij,ij->i", M, M)))
    L = max(spectral * mask_norm, 1e-12)
    Y = Z.copy()
    momentum = 1.0
    eps = np.finfo(X.dtype).eps
    for _ in range(max(8, iters)):
        for restart in range(2):
            ry = _frozen_residual(X, Y, M, y)
            grad = X.T @ (M * ry[:, None])
            ry64 = np.asarray(ry, dtype=np.float64)
            smooth_y = 0.5 * float(ry64 @ ry64)
            L = max(L * 0.8, 1e-12)
            for _line in range(60):
                candidate = Y - grad / L
                norms = np.linalg.norm(candidate, axis=0)
                candidate *= np.maximum(0, 1 - (lam / L) / np.maximum(norms, 1e-30))
                rc = _frozen_residual(X, candidate, M, y)
                rc64 = np.asarray(rc, dtype=np.float64)
                smooth = 0.5 * float(rc64 @ rc64)
                delta64 = np.asarray(candidate - Y, dtype=np.float64)
                majorizer = smooth_y + float(np.sum(grad * delta64)) + 0.5 * L * float(np.sum(delta64**2))
                # Roundoff allowance scales with the precision of the proposal.
                if smooth <= majorizer + 16 * eps * max(1, abs(smooth_y)):
                    break
                L *= 2
            else:
                return Z, objective
            candidate_objective = _group_objective(candidate, rc, lam)
            if candidate_objective <= objective:
                break
            # Re-evaluate a proximal step from the incumbent rather than
            # accepting an increase caused by extrapolation.
            Y = Z.copy()
            momentum = 1.0
        if candidate_objective > objective:
            break
        decrease = objective - candidate_objective
        next_momentum = 0.5 * (1 + np.sqrt(1 + 4 * momentum**2))
        Y = candidate + ((momentum - 1) / next_momentum) * (candidate - Z)
        Z, r, objective = candidate, rc, candidate_objective
        momentum = next_momentum
        if decrease <= tol * max(1, abs(objective)):
            break
    return Z, _group_objective(Z, _frozen_residual(X, Z, M, y), lam)


def _signed_relu_value_grad(V, X, y, signs, lam):
    """Actual-ReLU objective and gradient for fixed output signs.

    ``V[:, j]`` absorbs the nonnegative output magnitude; its orientation is
    the input direction, including when the fixed output sign is negative.
    The gradient uses the zero subgradient at zero activations and blocks.
    """
    activation = X @ V
    mask = activation > 0
    np.maximum(activation, 0, out=activation)
    residual = activation @ signs - y
    norms = np.linalg.norm(V, axis=0)
    value = 0.5 * float(residual @ residual) + lam * float(norms.sum())
    gradient = X.T @ (mask * (residual[:, None] * signs[None, :]))
    if lam:
        normalized = np.divide(V, norms[None, :], out=np.zeros_like(V),
                               where=norms[None, :] > 0)
        gradient += lam * normalized
    return value, gradient


def _make_device_relu_evaluator(X, y, signs, lam, device, dtype="float32"):
    """Cache a Torch sketch and return a NumPy value/gradient callback.

    The sketch is transferred once. Each evaluation transfers only the small
    parameter matrix and one packed value/gradient result. Products use the
    requested dtype; objective and group-norm reductions use float64.
    """
    if dtype not in ("float32", "float64"):
        raise ValueError("device refinement dtype must be 'float32' or 'float64'")
    import torch

    dt = torch.float32 if dtype == "float32" else torch.float64
    host_X = as_float32(X) if dtype == "float32" else np.asarray(X, dtype=np.float64)
    host_y = as_float32(y) if dtype == "float32" else np.asarray(y, dtype=np.float64)
    Xd = torch.as_tensor(host_X, dtype=dt, device=device)
    yd = torch.as_tensor(host_y, dtype=dt, device=device)
    sd = torch.as_tensor(np.asarray(signs), dtype=dt, device=device)

    def evaluate(V):
        with torch.no_grad():
            Vd = torch.as_tensor(np.asarray(V), dtype=dt, device=device)
            activation = Xd @ Vd
            mask = activation > 0
            activation.clamp_min_(0)
            residual = activation @ sd - yd
            gradient = Xd.T @ (mask * (residual[:, None] * sd[None, :]))
            V64 = Vd.to(torch.float64)
            norms = torch.linalg.vector_norm(V64, dim=0)
            r64 = residual.to(torch.float64)
            value = 0.5 * torch.dot(r64, r64) + lam * norms.sum()
            if lam:
                normalized = V64 / norms.clamp_min(torch.finfo(torch.float64).tiny)
                gradient += (lam * normalized).to(dt)
            packed = torch.cat((value.reshape(1), gradient.to(torch.float64).ravel()))
            host = packed.cpu().numpy()
            return float(host[0]), host[1:].reshape(V.shape)

    return evaluate


def _refine_relu_lbfgs(X, y, U, beta, lam, iters=120, tol=1e-9, device=None,
                       device_dtype="float32"):
    """Refine actual ReLU vectors with a limited-memory quasi-Newton proposal.

    This nonconvex local search crosses activation boundaries. It uses no
    per-atom Gram matrices. The caller still refits the finite dictionary on
    all data and checks its objective before replacing the incumbent.
    """
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise RuntimeError("L-BFGS refinement requires scipy") from exc
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    initial = U * np.abs(beta)[None, :]
    signs = np.sign(beta)
    signs[signs == 0] = 1
    evaluate = (_make_device_relu_evaluator(X, y, signs, lam, device, dtype=device_dtype)
                if device is not None
                else lambda V: _signed_relu_value_grad(V, X, y, signs, lam))
    best = initial.copy()
    best_value, initial_gradient = evaluate(best)
    if not np.isfinite(best_value) or not np.isfinite(initial_gradient).all():
        # A lower-precision proposal may not represent this initial model.
        # Return its unchanged float64 vectors for the caller's exact refit.
        return best, _signed_relu_value_grad(best, X, y, signs, lam)[0]
    if not initial.size or iters <= 0:
        return best, best_value

    def value_grad(flat):
        nonlocal best, best_value
        V = flat.reshape(initial.shape)
        with np.errstate(over="ignore", invalid="ignore"):
            value, gradient = evaluate(V)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            # Reject an overflowing line-search trial, retaining the finite
            # incumbent rather than returning invalid proposal directions.
            return float("inf"), np.zeros_like(flat)
        if value < best_value:
            best, best_value = V.copy(), value
        return value, gradient.ravel()

    minimize(value_grad, initial.ravel(), method="L-BFGS-B", jac=True,
             options={"maxiter": int(iters), "maxfun": max(20, 20 * int(iters)),
                      "maxls": 20, "maxcor": 10, "ftol": tol, "gtol": tol})
    return best, best_value


def propose_consolidation(
    engine,
    U_active,
    beta_active,
    lam,
    sketch_m=32768,
    rng=None,
    iters=120,
    rng_seed=0,
    row_norms=None,
    method="coordinate",
):
    """Host-sketch refinement, followed by projection to unit directions.

    ``coordinate`` and ``accelerated`` use frozen-mask group Lasso. ``lbfgs``
    directly refines the actual signed ReLU model with SciPy. ``lbfgs-device``
    evaluates that local search on a cached Torch CUDA sketch. All paths
    return directions only; the caller performs a full-data coefficient refit
    and objective comparison before accepting them.
    """
    rng = rng or np.random.default_rng(rng_seed)
    n, _d = engine.n, engine.d
    # honor the requested sketch size exactly (the caller's self-tuner owns the
    # budget); only fall back to full rows when the request reaches n
    m = min(sketch_m, n) if sketch_m else n
    if m < n:
        # Importance reweighting can inflate row scales and slow the proximal
        # proposal.  Use a uniform, unweighted objective subsample and rescale
        # lam to the subsample size; full-data master work follows in the caller.
        idx = rng.choice(n, size=m, replace=False)
        Xs, ys = engine.gather_rows(idx)
        row_w = None
        lam = lam * (m / n)
    else:
        Xs, ys = engine._Xhost, engine._yhost
        row_w = None
    if method in ("lbfgs", "lbfgs-device"):
        device = None
        device_dtype = "float32"
        if method == "lbfgs-device":
            if engine.backend != "torch":
                raise ValueError("lbfgs-device refinement requires a Torch CUDA engine")
            import torch

            device = next((dev for dev in engine.devices
                           if torch.device(dev).type == "cuda"), None)
            if device is None:
                raise ValueError("lbfgs-device refinement requires a Torch CUDA engine")
            device_dtype = "float64" if engine._dt == torch.float64 else "float32"
        Zc, _ = _refine_relu_lbfgs(Xs, ys, U_active, beta_active, lam,
                                  iters=iters, device=device, device_dtype=device_dtype)
    else:
        Xs = as_float32(Xs)
        ys = as_float32(ys)
        Ua = np.asarray(U_active, np.float32)
        M = (Xs @ Ua >= 0).astype(np.float32)
        Z0 = Ua * np.asarray(beta_active, np.float32)[None, :]
        Z, _ = gated_group_lasso(
            Xs, ys, M, lam, Z0=Z0, iters=iters, row_w=row_w, rng_seed=rng_seed,
            method=method,
        )
        Zc = Z.astype(np.float64)
    nz = np.linalg.norm(Zc, axis=0)
    keep = np.where(nz > 1e-10 * max(1.0, float(nz.max()) if nz.size else 1.0))[0]
    if not len(keep):
        return None
    keep = keep[np.argsort(-nz[keep])]  # strongest blocks first
    # Frozen-mask z_j includes the output sign; actual-ReLU v_j does not.
    # Recover the input orientation before the caller's full-data refit.
    signs = np.ones(len(keep)) if method in ("lbfgs", "lbfgs-device") else np.sign(
        np.asarray(beta_active, dtype=np.float64)[keep]
    )
    signs[signs == 0] = 1.0
    return (Zc[:, keep] * signs[None, :]) / nz[keep][None, :]
