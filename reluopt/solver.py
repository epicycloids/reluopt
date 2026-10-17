"""Column generation for scalar-output two-layer ReLU fitting."""

from __future__ import annotations

import time
from numbers import Real

import numpy as np

from .adaptive_pricing import search_signed_atoms
from .consolidation import propose_consolidation
from .data_engine import ShardedData
from .model import ReLUModel
from .pricing_bounds import dual_gap, estimate_pricing_bounds
from .restricted_master import AtomLasso


def _validate_options(**options):
    positive_ints = {"add_per_round", "max_cols", "patience"}
    nonnegative_ints = {
        "max_iter", "certify_every", "cert_steps", "gated_every", "gated_iters", "max_feature_bytes"
    }
    for name in positive_ints | nonnegative_ints:
        value = options[name]
        if name == "max_feature_bytes" and value is None:
            continue
        lower = 1 if name in positive_ints else 0
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < lower
        ):
            raise ValueError(f"{name} must be an integer >= {lower}")
    for name in ("lam", "eps_rel", "eps_heur", "price_tol", "stall_rtol"):
        value = options[name]
        if (
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0
        ):
            raise ValueError(f"{name} must be finite and nonnegative")
    budget = options["time_budget"]
    if budget is not None and (
        isinstance(budget, (bool, np.bool_)) or not isinstance(budget, Real)
        or not np.isfinite(budget) or budget <= 0
    ):
        raise ValueError("time_budget must be finite and positive")
    sketch = options["sketch_m"]
    if not isinstance(sketch, str) or sketch != "auto":
        if (
            isinstance(sketch, (bool, np.bool_))
            or not isinstance(sketch, (int, np.integer))
            or sketch < 0
        ):
            raise ValueError("sketch_m must be 'auto' or a nonnegative integer")


def _dedup_cols(Unew_cols, U_existing, cos_tol=0.9995):
    return _dedup_cols_indexed(Unew_cols, U_existing, cos_tol)[0]


def _dedup_cols_indexed(Unew_cols, U_existing, cos_tol=0.9995, normalize=True):
    """Oriented deduplication plus indices into ``Unew_cols``."""
    fresh = []
    indices = []
    for index, u in enumerate(Unew_cols):
        u = np.asarray(u, float)
        if normalize:
            u = u / (np.linalg.norm(u) + 1e-30)
        else:
            # Retained oracle features were formed from this exact vector.
            # The oracle has already normalized it; changing its scale here
            # would misalign the cached column and the direction stored by the
            # restricted master.
            u = u.copy()
        if (
            U_existing is not None
            and U_existing.shape[1]
            and float(np.max(u @ U_existing)) > cos_tol
        ):
            continue
        if any(float(u @ w) > cos_tol for w in fresh):
            continue
        fresh.append(u)
        indices.append(index)
    return fresh, indices


def make_engine(
    X, y, device="cpu", devices=None, dtype=None, threads=False, weights=None
):
    """Build the data engine.  device: "cpu" (numpy), "torch" / "cuda" /
    "cuda:0,cuda:1" (torch backend; ROCm presents as cuda)."""
    if devices is not None:
        backend = (
            "torch" if any("cuda" in d or "torch" in d for d in devices) else "numpy"
        )
        devs = [d.replace("torch:", "") for d in devices]
    elif device in ("cpu", "numpy"):
        backend, devs = "numpy", ["cpu"]
    elif device.startswith(("cuda", "torch")):
        backend = "torch"
        spec = device.replace("torch:", "")
        devs = spec.split(",") if "," in spec else [spec if spec != "torch" else "cuda"]
    else:
        raise ValueError(f"unknown device: {device!r}")
    if dtype is None:
        dtype = "float32" if backend == "torch" else "float64"
    return ShardedData(
        X,
        y,
        devices=devs,
        backend=backend,
        dtype=dtype,
        threads=threads,
        weights=weights,
    )


def _prepare_initial_atoms(initial_U, initial_beta, n_features):
    """Normalize a supplied network without changing its represented function."""
    if initial_U is None:
        if initial_beta is not None:
            raise ValueError("initial_beta requires initial_U")
        return np.empty((n_features, 0)), np.empty(0), 0
    if np.iscomplexobj(initial_U) or np.iscomplexobj(initial_beta):
        raise ValueError("initial parameters must be real")
    U = np.array(initial_U, dtype=np.float64, copy=True)
    if U.ndim != 2 or U.shape[0] != n_features or not np.all(np.isfinite(U)):
        raise ValueError("initial_U must be a finite (n_features, n_atoms) array")
    width = U.shape[1]
    beta = (
        np.zeros(width) if initial_beta is None
        else np.array(initial_beta, dtype=np.float64, copy=True)
    )
    if beta.shape != (width,) or not np.all(np.isfinite(beta)):
        raise ValueError("initial_beta must be a finite vector matching initial_U")
    # Scale first so finite directions near the floating-point limits can still
    # be normalized. Exact-zero directions represent the zero function.
    scales = np.max(np.abs(U), axis=0)
    keep = scales != 0
    U = U[:, keep]
    scales = scales[keep]
    U /= scales
    norms = np.linalg.norm(U, axis=0)
    U /= norms
    with np.errstate(over="ignore", invalid="ignore"):
        beta = (beta[keep] * scales) * norms
    if not np.all(np.isfinite(beta)):
        raise ValueError("normalized initial coefficients exceed float64 range")
    return U, beta, width


def _initial_objective(eng, U, beta, lam):
    """Evaluate an incumbent against original observations in float64."""
    residual = eng._yhost - ReLUModel(U, beta).predict(
        eng._Xhost, chunk_rows=eng.chunk
    )
    return 0.5 * float(residual @ residual) + lam * float(np.abs(beta).sum())


def solve_rcg(
    X,
    y,
    lam,
    device="cpu",
    engine=None,
    dtype_data=None,
    max_iter=100,
    eps_rel=1e-4,
    eps_heur=1e-3,
    price_tol=1e-3,
    add_per_round=8,
    max_cols=512,
    certify_every=5,
    cert_steps=15,
    gated_every=1,
    gated_iters=100,
    sketch_m="auto",
    time_budget=None,
    rng_seed=0,
    verbose=False,
    stall_rtol=1e-7,
    patience=6,
    final_verify=True,
    devices=None,
    threads=False,
    oracle_kwargs=None,
    allow_numerical_bound_stop=False,
    max_feature_bytes=None,
    gated_method="auto",
    final_diagnostics=True,
    initial_U=None,
    initial_beta=None,
):
    """Run RCG.  Returns dict(obj, nu, beta, U, n_active, gap_cert,
    gap_heur, rho_lb, history, timings, ...).

    Despite its concise name, ``gap_cert`` is computed with round-to-nearest
    arithmetic and is not outward-certified.  It does not
    trigger stopping unless ``allow_numerical_bound_stop=True``.

    ``final_diagnostics=False`` skips the final pricing search and bounds.
    Original-data float64 reconstruction and finite-master verification still
    run when ``final_verify=True`` (the default).

    ``initial_U`` has shape ``(n_features, n_atoms)``. Nonzero directions are
    normalized and their norms absorbed into ``initial_beta``; omitted
    coefficients start at zero. The initial finite dictionary is refitted in
    float64 before search. Its fit is retained if subsequent search loses on
    the original-data objective. Directions and their activation chambers may
    change during search, and new directions may be added.

    ``max_iter=0`` performs only the coefficient refit, without pricing or
    consolidation. Initial refitting and final verification are mandatory
    work outside the soft search deadline. A column or feature budget smaller
    than the nonzero initial dictionary raises ``ValueError``.
    """
    if gated_method not in {"auto", "coordinate", "accelerated", "lbfgs", "lbfgs-device"}:
        raise ValueError("unknown gated_method")
    _validate_options(
        lam=lam, max_iter=max_iter, eps_rel=eps_rel, eps_heur=eps_heur,
        price_tol=price_tol, add_per_round=add_per_round, max_cols=max_cols,
        certify_every=certify_every, cert_steps=cert_steps,
        gated_every=gated_every, gated_iters=gated_iters, sketch_m=sketch_m,
        time_budget=time_budget, stall_rtol=stall_rtol, patience=patience,
        max_feature_bytes=max_feature_bytes,
    )
    t_start = time.perf_counter()
    eng = engine or make_engine(
        X, y, device=device, devices=devices, dtype=dtype_data, threads=threads
    )
    n, _d = eng.n, eng.d
    requested_max_cols = max_cols
    if max_feature_bytes is not None:
        max_cols = min(max_cols, max_feature_bytes // eng.feature_buffer_bytes(1))
    initial_dirs, initial_coefficients, initial_supplied = _prepare_initial_atoms(
        initial_U, initial_beta, eng.d
    )
    if initial_dirs.shape[1] > max_cols:
        raise ValueError("initial dictionary exceeds the column or feature budget")
    y64 = eng._yhost
    rng = np.random.default_rng(rng_seed)
    if sketch_m == "auto":
        sketch_m = 0 if n <= 65536 else 32768
    ok = dict(sketch_m=sketch_m or 0)
    if oracle_kwargs:
        ok.update(oracle_kwargs)
    row_norms = np.linalg.norm(np.asarray(eng._Xhost, dtype=np.float64), axis=1)

    master = AtomLasso(eng)
    prev_dirs = []
    cert_warm = None
    gated_m = sketch_m  # self-tuning gated sketch size (see below)
    gated_ok = True
    hist = {
        k: []
        for k in (
            "iter",
            "n_cols",
            "n_active",
            "obj",
            "rho_lb",
            "ub",
            "gap_cert",
            "gap_heur",
            "time",
        )
    }
    timings = {"master": 0.0, "gated": 0.0, "price": 0.0, "cert": 0.0, "other": 0.0}
    feature_reuse = {
        "pricing_calls": 0,
        "requested_calls": 0,
        "retained_calls": 0,
        "fallback_calls": 0,
        "retained_payload_bytes_peak": 0,
        "required_peak_payload_bytes_peak": 0,
        "appended_reused_columns": 0,
        "appended_rebuilt_columns": 0,
    }
    escalated = False
    stop_reason = "max_iter"
    obj = 0.5 * float(y64 @ y64)
    nu_sh = eng.ys
    initial_fit = None
    initialization = None
    if initial_U is not None:
        t0 = time.perf_counter()
        initial_obj = _initial_objective(eng, initial_dirs, initial_coefficients, lam)
        initial_fit = verify_solution(
            eng, lam, initial_dirs, beta_start=initial_coefficients,
            run_diagnostics=False,
        )
        initial_fit["obj"] = _initial_objective(
            eng, initial_dirs, initial_fit["beta"], lam
        )
        refit_seconds = time.perf_counter() - t0
        if max_iter:
            master.set_atoms(initial_dirs)
            master.beta = initial_fit["beta"].copy()
            master.S = np.flatnonzero(master.beta != 0).tolist()
            # The device dictionary can differ from its float64 replay.
            # Correct that finite fit before using its residual for pricing.
            _, nu_sh, obj, _ = master.solve(lam)
        timings["master"] += time.perf_counter() - t0
        initialization = dict(
            n_supplied=initial_supplied,
            n_cols=initial_dirs.shape[1],
            obj=initial_obj,
            refit_obj=initial_fit["obj"],
            refit_seconds=refit_seconds,
            fallback_used=False,
        )
    elif max_iter == 0:
        _, nu_sh, obj, _ = master.solve(lam)

    for it in range(max_iter):
        if time_budget and time.perf_counter() - t_start >= time_budget:
            stop_reason = "time_budget"
            break
        # ---------------- master ----------------
        t0 = time.perf_counter()
        beta, nu_sh, obj, S = master.solve(lam)
        timings["master"] += time.perf_counter() - t0

        # ------- gated consolidation: REPLACE working set if it wins --------
        # at the sketch-size cap with a losing proposal (typical on agnostic
        # real data), halve the gated frequency instead of growing further
        gated_skip = (
            (not gated_ok)
            and sketch_m
            and gated_m >= 4 * sketch_m
            and it % (2 * gated_every) != 0
        )
        if gated_every and len(S) >= 2 and it % gated_every == 0 and not gated_skip:
            t0 = time.perf_counter()
            Sg = S
            if len(Sg) > 96:  # consolidate the strongest blocks only
                ord_ = np.argsort(-np.abs(master.beta[np.asarray(S)]))[:96]
                Sg = [S[i] for i in ord_]
            method = gated_method
            if method == "auto":
                from .restricted_master import _solve_triangular
                if _solve_triangular is not None and len(Sg) >= 4:
                    method = "lbfgs"
                    proposal_rows = min(gated_m, n) if gated_m else n
                    if (eng.backend == "torch" and any("cuda" in dev for dev in eng.devices)
                            and str(eng._dt).endswith("float32")
                            and proposal_rows * eng.d * len(Sg) >= 1_000_000):
                        method = "lbfgs-device"
                else:
                    method = "accelerated" if eng.d >= 32 or len(Sg) >= 8 else "coordinate"
            Un = propose_consolidation(
                eng,
                master.U[:, Sg],
                master.beta[np.asarray(Sg)],
                lam,
                sketch_m=gated_m,
                rng=rng,
                iters=gated_iters,
                rng_seed=rng_seed + it,
                row_norms=row_norms,
                method=method,
            )
            if Un is not None and Un.shape[1] >= 1:
                trial = AtomLasso(eng)
                trial.set_atoms(Un)
                # warm start: the proposal's blocks are the expected active set
                # (positive coefficients by construction of the projection)
                trial.S = list(range(Un.shape[1]))
                trial.beta = np.full(Un.shape[1], 1e-12)
                bt, nut, objt, St = trial.solve(lam)
                if objt <= obj:
                    master = trial
                    beta, nu_sh, obj, S = bt, nut, objt, St
                    gated_ok = True
                    if gated_m:  # proposal won: decay sketch toward baseline
                        gated_m = max(sketch_m, gated_m // 2)
                else:
                    # proposal lost: usually the sketch was too noisy at this n
                    # -- the safeguard detects exactly that, so GROW the gated
                    # sketch (self-tuning); append only the strongest blocks
                    gated_ok = False
                    if gated_m and gated_m < min(n, 4 * sketch_m):
                        gated_m = min(n, 4 * sketch_m, 4 * gated_m)
                    fresh = _dedup_cols(list(Un.T[:8]), master.U)
                    fresh = fresh[: max(0, max_cols - master.P)]
                    if fresh:
                        master.append_atoms(np.stack(fresh, axis=1))
                        beta, nu_sh, obj, S = master.solve(lam)
            timings["gated"] += time.perf_counter() - t0

        # ---------------- prune zero-weight atoms; same-chamber merge -------
        t0 = time.perf_counter()
        bc = master.beta
        # Removing even a tiny nonzero coefficient can increase the objective.
        keep = np.flatnonzero(bc != 0).tolist()
        if len(keep) < master.P:
            master.keep(keep)
            _, nu_sh, obj, S = master.solve(lam)
        if it % 3 == 2 and len(master.S) >= 2:
            trial = _compress_chambers(master, eng, lam, obj)
            if trial is not None:
                master = trial
                beta, nu_sh, obj, S = master.solve(lam)
        n_active = len(S)
        timings["other"] += time.perf_counter() - t0
        if not master.converged:
            # Correct the current finite fit before introducing more atoms.
            stop_reason = "master_iteration_limit"
            values = dict(
                iter=it, n_cols=master.P, n_active=n_active, obj=obj,
                rho_lb=0.0, ub=None, gap_cert=None, gap_heur=None,
                time=time.perf_counter() - t_start,
            )
            for key, value in values.items():
                hist[key].append(value)
            break

        # ---------------- pricing (joint +-nu) ----------------
        t0 = time.perf_counter()
        nu_host = eng.gather_vec(nu_sh)
        Ua = master.U[:, S] if n_active else None
        ba = master.beta[np.asarray(S)] if n_active else None
        # regimes with many active atoms need bigger batches, but
        # only while consolidation is keeping up (else batching feeds a runaway)
        eff_add = (
            max(add_per_round, min(32, n_active // 4)) if gated_ok else add_per_round
        )
        price_ok = dict(ok)
        # No append follows the last permitted round, so retaining finalists
        # there could only create an unused buffer.
        if it + 1 >= max_iter or master.P >= max_cols:
            price_ok["retain_finalist_features"] = False
        pr = search_signed_atoms(
            eng,
            nu_sh,
            nu_host,
            lam,
            k=eff_add,
            rng=rng,
            U_active=Ua,
            beta_active=ba,
            prev_dirs=prev_dirs,
            use_sdp=escalated,
            row_norms=row_norms,
            **price_ok,
        )
        cand = list(pr["plus"]) + list(pr["minus"])
        addition = pr.get("addition", [(v, v, u) for v, u in cand])
        addition_cache = pr.get("addition_cache")
        reuse = pr.get("feature_reuse", {})
        feature_reuse["pricing_calls"] += 1
        if reuse.get("requested"):
            feature_reuse["requested_calls"] += 1
            if reuse.get("retained"):
                feature_reuse["retained_calls"] += 1
            else:
                feature_reuse["fallback_calls"] += 1
        feature_reuse["retained_payload_bytes_peak"] = max(
            feature_reuse["retained_payload_bytes_peak"],
            int(reuse.get("retained_payload_bytes", 0)),
        )
        feature_reuse["required_peak_payload_bytes_peak"] = max(
            feature_reuse["required_peak_payload_bytes_peak"],
            int(reuse.get("required_peak_payload_bytes", 0)),
        )
        rho_lb = max([v for v, _ in cand], default=0.0)
        prev_dirs = [u for _, u in sorted(cand, key=lambda z: -z[0])[:6]]
        # Keep only the compact append payload.  In particular, do not retain
        # the complete oracle result while certificates or the next pricing
        # call allocate their own finalist buffers.
        pr = None
        timings["price"] += time.perf_counter() - t0

        # ---------------- certificate (periodic; honest reporting) ----------
        no_violator = rho_lb <= lam * (1.0 + price_tol)
        if no_violator:
            # This round cannot append, even when it triggers the one-time
            # escalated recheck, so release any unused retained columns before
            # certificate work allocates its own buffers.
            addition_cache = None
        ub = gap_cert = None
        if certify_every and (
            it % certify_every == certify_every - 1 or no_violator or it == max_iter - 1
        ):
            t0 = time.perf_counter()
            cert = estimate_pricing_bounds(
                eng,
                nu_host,
                n_steps=cert_steps,
                warm=cert_warm,
                sketch_m=sketch_m or 0,
                rng=rng,
                row_norms=row_norms,
            )
            cert_warm = cert["warm"]
            ub = max(cert["ub_plus"], cert["ub_minus"])
            gap_cert, _ = dual_gap(
                y64, nu_host, lam, cert["ub_plus"], cert["ub_minus"], obj
            )
            timings["cert"] += time.perf_counter() - t0
        gap_heur, _ = dual_gap(
            y64, nu_host, lam, max(rho_lb, lam), max(rho_lb, lam), obj
        )

        hist["iter"].append(it)
        hist["n_cols"].append(master.P)
        hist["n_active"].append(n_active)
        hist["obj"].append(obj)
        hist["rho_lb"].append(float(rho_lb))
        hist["ub"].append(None if ub is None else float(ub))
        hist["gap_cert"].append(None if gap_cert is None else float(gap_cert))
        hist["gap_heur"].append(float(gap_heur))
        hist["time"].append(time.perf_counter() - t_start)
        if verbose:
            ubs = f"{ub:.4f}" if ub is not None else "  --  "
            gcs = f"{gap_cert:.3e}" if gap_cert is not None else "   --   "
            print(
                f"[RCG {it:03d}] cols={master.P:4d} act={n_active:3d} "
                f"obj={obj:.6f} rho_lb={rho_lb:.4f} ub={ubs} "
                f"gapN={gcs} gapH={gap_heur:.3e}"
            )

        # ---------------- stopping ----------------
        if (
            allow_numerical_bound_stop
            and gap_cert is not None
            and gap_cert <= eps_rel * max(1.0, abs(obj))
        ):
            addition_cache = None
            stop_reason = "numerical_bound"
            break
        if eps_heur and gap_heur <= eps_heur * max(1.0, abs(obj)) and it > 2:
            addition_cache = None
            stop_reason = "heuristic_gap"
            break
        if no_violator:
            addition_cache = None
            if not escalated:
                escalated = True
                continue
            stop_reason = "no_violator"
            break
        if escalated and not no_violator:
            escalated = False
        if len(hist["obj"]) > patience:
            rec = hist["obj"][-(patience + 1) :]
            if (
                0 <= (rec[0] - rec[-1]) <= stall_rtol * max(1.0, abs(rec[-1]))
                and rho_lb <= lam * 1.05
            ):
                addition_cache = None
                stop_reason = "plateau"
                break
        if time_budget and time.perf_counter() - t_start > time_budget:
            addition_cache = None
            stop_reason = "time_budget"
            break

        # Appending after the final permitted round creates columns that are
        # never solved or scored and can make the returned model exceed the
        # declared column cap.
        if it + 1 >= max_iter:
            addition_cache = None
            break

        # ---------------- add violating atoms ----------------
        ranked_additions = sorted(enumerate(addition), key=lambda item: -item[1][0])
        viol = [
            (index, item[2])
            for index, item in ranked_additions
            if item[1] > lam * (1.0 + 1e-4)
        ][: 2 * eff_add]
        fresh, fresh_positions = _dedup_cols_indexed(
            # search_signed_atoms returns normalized directions.  Preserve those exact
            # bytes in both paths so enabling reuse changes only data movement.
            [u for _, u in viol],
            master.U,
            normalize=False,
        )
        if not fresh and not no_violator:
            addition_cache = None
            stop_reason = "no_new_column"
            break
        if fresh:
            slots = max_cols - master.P
            if slots <= 0:
                addition_cache = None
                stop_reason = (
                    "feature_budget" if max_cols < requested_max_cols else "column_budget"
                )
                break
            fresh = fresh[:slots]
            fresh_positions = fresh_positions[:slots]
            Ufresh = np.stack(fresh, axis=1)
            cache_indices = [viol[pos][0] for pos in fresh_positions]
            if addition_cache is not None:
                Anew = eng.slice_cols(addition_cache["A_sh"], cache_indices)
                master.append_atoms(
                    Ufresh,
                    A_sh=Anew,
                    aty=np.asarray(addition_cache["aty"])[cache_indices],
                    sqnorms=np.asarray(addition_cache["sqnorm"])[cache_indices],
                    dual_corr=np.asarray(addition_cache["dual_corr"])[cache_indices],
                )
                feature_reuse["appended_reused_columns"] += len(fresh)
                Anew = None
            else:
                master.append_atoms(Ufresh)
                feature_reuse["appended_rebuilt_columns"] += len(fresh)
        addition_cache = None

    # ---------------- final float64 diagnostic replay ----------------
    out = dict(
        history=hist,
        timings=timings,
        stop_reason=stop_reason,
        n_iter=len(hist["iter"]),
        feature_reuse=feature_reuse,
        max_cols=max_cols,
        max_feature_bytes=max_feature_bytes,
        feature_payload_bytes=eng.feature_buffer_bytes(master.P),
        diagnostics_computed=False,
        initialization=initialization,
    )
    if max_iter == 0 and initial_fit is not None:
        # Reuse the original-data refit rather than rebuilding it a second
        # time. Coefficient-only fitting never invokes a pricing diagnostic.
        out.update(initial_fit)
    elif final_verify:
        t0 = time.perf_counter()
        ver = verify_solution(
            eng,
            lam,
            master.U,
            cert_steps=max(30, 2 * cert_steps),
            rng=rng,
            sketch_m=sketch_m or 0,
            row_norms=row_norms,
            beta_start=master.beta,
            # A zero-column stop has no final diagnostics in the default
            # solver path, but its residual/objective/KKT still get replayed.
            run_diagnostics=final_diagnostics and bool(master.P) and bool(max_iter),
        )
        out.update(ver)
        timings["verify"] = time.perf_counter() - t0
    else:
        out.update(
            dict(
                obj=obj,
                nu=eng.gather_vec(nu_sh) if master.P else y64,
                beta=master.beta,
                U=master.U,
                n_active=len(master.S),
                n_cols=master.P,
                rho_lb=hist["rho_lb"][-1] if hist["rho_lb"] else 0.0,
                gap_cert=hist["gap_cert"][-1] if hist["gap_cert"] else None,
                gap_heur=hist["gap_heur"][-1] if hist["gap_heur"] else None,
                finite_master_kkt=master.kkt_violation,
                finite_master_converged=master.converged,
            )
        )
        if not final_diagnostics:
            out.update(
                rho_lb=None, ub_plus=None, ub_minus=None,
                gap_cert=None, gap_heur=None, theta=None,
            )
    if initial_fit is not None and max_iter:
        final_obj = _initial_objective(eng, out["U"], out["beta"], lam)
        if final_obj > initial_fit["obj"]:
            # Stored float32 features and a replaced working set may appear to
            # improve while their actual float64 ReLU model loses. Preserve
            # the verified initial fit in that case.
            fallback = initial_fit
            if final_diagnostics and max_iter and initial_fit["n_cols"]:
                t0 = time.perf_counter()
                fallback = verify_solution(
                    eng, lam, initial_dirs, beta_start=initial_fit["beta"],
                    cert_steps=max(30, 2 * cert_steps), rng=rng,
                    sketch_m=sketch_m or 0, row_norms=row_norms,
                )
                timings["verify"] = timings.get("verify", 0.0) + time.perf_counter() - t0
            out.update(fallback)
            initialization["fallback_used"] = True
    out["feature_payload_bytes"] = eng.feature_buffer_bytes(out["n_cols"])
    out["model"] = ReLUModel(out["U"], out["beta"])
    out["verified_float64"] = bool(
        final_verify or max_iter == 0 and initial_fit is not None
        or initialization is not None and initialization["fallback_used"]
    )
    out["time"] = time.perf_counter() - t_start
    out["gap_cert_is_rigorous"] = False
    out["certificate_status"] = (
        "exact-arithmetic bound evaluated in round-to-nearest floating point; "
        "not outward-certified"
        if out["diagnostics_computed"] else "not_computed"
    )
    return out


def _compress_chambers(master, eng, lam, incumbent_obj):
    """Same-chamber, same-sign merge over active finite-master atoms."""
    S = master.S
    if len(S) < 2:
        return None
    keys = eng.chamber_keys(master.U[:, S])
    beta = master.beta
    groups = {}
    for t, j in enumerate(S):
        sgn = 1.0 if beta[j] >= 0 else -1.0
        groups.setdefault((keys[t], sgn), []).append(j)
    merged = {k: v for k, v in groups.items() if len(v) > 1}
    if not merged:
        return None
    drop = set()
    new_dirs = []
    new_beta = []
    for (_, sgn), js in merged.items():
        vG = sum(abs(beta[j]) * master.U[:, j] for j in js)
        nrm = np.linalg.norm(vG)
        if nrm > 0:
            new_dirs.append(vG / nrm)
            new_beta.append(sgn * nrm)
        drop.update(js)
    keep = [j for j in range(master.P) if j not in drop]
    U = master.U[:, keep]
    beta = master.beta[keep]
    if new_dirs:
        U = np.concatenate([U, np.stack(new_dirs, axis=1)], axis=1)
        beta = np.r_[beta, new_beta]
    trial = AtomLasso(eng)
    trial.set_atoms(U)
    trial.beta = beta
    trial.S = np.flatnonzero(beta != 0).tolist()
    _, _, obj, _ = trial.solve(lam)
    # Stored features can differ after projection/rounding, especially in
    # float32. Retain the original working set if the actual refit loses.
    return trial if obj <= incumbent_obj else None


def verify_solution(
    eng, lam, U_dirs, cert_steps=30, rng=None, sketch_m=0, row_norms=None,
    beta_start=None, run_diagnostics=True,
):
    """Stricter float64 diagnostic replay of a returned finite working set.

    The replay always rebuilds original-data float64 features and tightens the
    finite-master tolerance. With ``run_diagnostics=True`` it also escalates
    heuristic pricing, rescores all rows, and evaluates both bound forms. These
    optional diagnostics are not a global price oracle or an outward-certified
    verification.
    """
    rng = rng or np.random.default_rng(0)
    # Use original observations and float64 matrix products. Merely changing
    # reduction precision leaves float32 feature construction errors intact.
    eng = ShardedData(
        eng._Xhost, eng._yhost, backend="numpy", dtype="float64", chunk=eng.chunk
    )
    U = np.array(U_dirs, dtype=np.float64, copy=True)
    if U.ndim != 2 or U.shape[0] != eng.d or not np.all(np.isfinite(U)):
        raise ValueError("U_dirs must be a finite (n_features, n_atoms) array")
    if np.any(np.linalg.norm(U, axis=0) > 1 + 32 * np.finfo(float).eps):
        raise ValueError("U_dirs columns must have Euclidean norm <= 1")
    master = AtomLasso(eng)
    master.set_atoms(U)
    if beta_start is not None:
        initial = np.asarray(beta_start, dtype=np.float64)
        if initial.shape != (U.shape[1],) or not np.all(np.isfinite(initial)):
            raise ValueError("beta_start must match the supplied directions")
        master.beta = initial.copy()
        master.S = np.flatnonzero(initial != 0).tolist()
    beta, nu_sh, obj, S = master.solve(lam, tol=1e-9)
    nu_host = eng.gather_vec(nu_sh)
    result = dict(
        obj=obj,
        nu=nu_host,
        beta=beta,
        U=U,
        n_active=len(S),
        n_cols=U.shape[1],
        rho_lb=None,
        ub_plus=None,
        ub_minus=None,
        gap_cert=None,
        gap_heur=None,
        theta=None,
        finite_master_kkt=master.kkt_violation,
        finite_master_converged=master.converged,
        diagnostics_computed=False,
    )
    if not run_diagnostics:
        return result
    pr = search_signed_atoms(
        eng,
        nu_sh,
        nu_host,
        lam,
        k=4,
        rng=rng,
        U_active=U[:, S] if S else None,
        sketch_m=sketch_m,
        n_random=32,
        iters_sketch=60,
        iters_full=15,
        top_polish=10,
        use_sdp=True,
        row_norms=row_norms,
    )
    rho_lb = pr["best"]
    cert = estimate_pricing_bounds(
        eng,
        nu_host,
        n_steps=cert_steps,
        forms=("joint", "split"),
        sketch_m=sketch_m,
        rng=rng,
        row_norms=row_norms,
    )
    y64 = eng._yhost
    gap_cert, theta = dual_gap(
        y64, nu_host, lam, cert["ub_plus"], cert["ub_minus"], obj
    )
    gap_heur, _ = dual_gap(y64, nu_host, lam, max(rho_lb, lam), max(rho_lb, lam), obj)
    result.update(
        rho_lb=float(rho_lb),
        ub_plus=float(cert["ub_plus"]),
        ub_minus=float(cert["ub_minus"]),
        gap_cert=float(gap_cert),
        gap_heur=float(gap_heur),
        theta=float(theta),
        diagnostics_computed=True,
    )
    return result
