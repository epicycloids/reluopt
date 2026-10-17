"""Warm-started Lasso over a finite dictionary of stored ReLU features.

The working-set solver caches float64 Gram columns and uses normalized
coordinates. Coordinate sweeps identify the support; Newton steps solve the
current orthant, stopping at the first sign crossing. All stopping decisions
use KKT residuals recomputed against the stored features, without adding a
ridge penalty to the objective.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.linalg import solve_triangular as _solve_triangular
except ImportError:  # SciPy is an optional performance dependency.
    _solve_triangular = None


class AtomLasso:
    """Working set of ReLU atoms plus warm-started numerical Lasso solves."""

    def __init__(self, engine, ridge=1e-10):
        self.eng = engine
        # Retained for compatibility: conditioning is handled by a least-squares
        # fallback, rather than perturbing the Lasso objective with a ridge.
        self.ridge = ridge
        self.kkt_violation = np.inf
        self.converged = False
        self.n_iter = 0
        self.stopping_reason = "not_solved"
        self.A = None  # sharded atom blocks
        self.U = np.zeros((engine.d, 0))  # host float64 unit directions
        self.Aty = np.zeros(0)
        self.norms = np.zeros(0)
        self.G = np.zeros((0, 0))
        self.known = np.zeros(0, dtype=bool)
        self._gram_rows = np.zeros(0, dtype=np.intp)
        self.S = []
        self.beta = np.zeros(0)
        self._invalidate_solution()

    def _invalidate_solution(self):
        self._cached_beta = None
        self._cached_residual = None
        self._cached_corr = None
        self._cached_rss = None
        self._cached_verified = False

    def _solution_matches(self):
        # beta remains publicly assignable for warm starts. Checking its value
        # also catches in-place writes without a property/setter API change.
        return self._cached_beta is not None and np.array_equal(
            self.beta, self._cached_beta
        )

    @property
    def P(self):
        return self.U.shape[1]

    def set_atoms(self, U, A_sh=None, aty=None, sqnorms=None):
        U = np.asarray(U, np.float64)
        self.A = A_sh if A_sh is not None else self.eng.build_atoms(U)
        self.U = U
        self._validate_prebuilt(self.A, U.shape[1])
        self.Aty = (
            self.eng.screen(self.A, self.eng.ys)
            if aty is None
            else self._validate_stats(aty, U.shape[1], "aty")
        )
        sq = (
            self.eng.sqnorms(self.A)
            if sqnorms is None
            else self._validate_stats(sqnorms, U.shape[1], "sqnorms")
        )
        self.norms = np.sqrt(np.maximum(sq, 1e-300))
        P = U.shape[1]
        self.G = np.zeros((P, P))
        self.known = np.zeros(P, dtype=bool)
        self._gram_rows = np.zeros(P, dtype=np.intp)
        self.S = []
        self.beta = np.zeros(P)
        self._invalidate_solution()

    def append_atoms(self, Unew, A_sh=None, aty=None, sqnorms=None, dual_corr=None):
        """Append directions, optionally reusing aligned full-data features.

        Omitting the optional arrays rebuilds the features from their
        directions. ``dual_corr`` may reuse ``Anew.T @ (y - Aold @ beta)``
        from full-data pricing of this same fitted model. Its use still
        requires a true residual/KKT verification before reporting convergence.
        """
        Unew = np.asarray(Unew, np.float64)
        if Unew.shape[1] == 0:
            return
        if self.P == 0:
            self.set_atoms(Unew, A_sh=A_sh, aty=aty, sqnorms=sqnorms)
            return
        Anew = self.eng.build_atoms(Unew) if A_sh is None else A_sh
        self._validate_prebuilt(Anew, Unew.shape[1])
        cached = self._solution_matches()
        self.A = self._concat(self.A, Anew)
        k = Unew.shape[1]
        P0 = self.P
        self.U = np.concatenate([self.U, Unew], axis=1)
        aty_new = (
            self.eng.screen(Anew, self.eng.ys)
            if aty is None
            else self._validate_stats(aty, k, "aty")
        )
        sq_new = (
            self.eng.sqnorms(Anew)
            if sqnorms is None
            else self._validate_stats(sqnorms, k, "sqnorms")
        )
        self.Aty = np.concatenate([self.Aty, aty_new])
        self.norms = np.concatenate([self.norms, np.sqrt(np.maximum(sq_new, 1e-300))])
        G = np.zeros((P0 + k, P0 + k))
        G[:P0, :P0] = self.G
        self.G = G
        self.known = np.zeros(P0 + k, dtype=bool)
        self._gram_rows = np.concatenate([self._gram_rows, np.zeros(k, dtype=np.intp)])
        self.beta = np.concatenate([self.beta, np.zeros(k)])
        if cached:
            # Appending zero coefficients leaves the fitted residual untouched.
            # Only correlations with the new feature block need a data pass.
            if dual_corr is None:
                new_corr = -self.eng.screen(Anew, self._cached_residual)
            else:
                new_corr = self._validate_stats(dual_corr, k, "dual_corr")
                self._cached_verified = False
            self._cached_corr = np.concatenate([self._cached_corr, new_corr])
            self._cached_beta = self.beta.copy()
        else:
            self._invalidate_solution()

    def _validate_prebuilt(self, A_sh, width):
        if A_sh is None or len(A_sh) != len(self.eng.slices):
            raise ValueError("prebuilt atom shards do not match the engine")
        for A, (start, end) in zip(A_sh, self.eng.slices):
            if tuple(A.shape) != (end - start, width):
                raise ValueError("prebuilt atom shard has the wrong shape")

    @staticmethod
    def _validate_stats(values, width, name):
        values = np.asarray(values, np.float64)
        if values.shape != (width,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a finite vector of length {width}")
        return values

    def _concat(self, A_sh, B_sh):
        if self.eng.backend == "numpy":
            return [np.concatenate([a, b], axis=1) for a, b in zip(A_sh, B_sh)]
        import torch

        return [torch.cat([a, b], dim=1) for a, b in zip(A_sh, B_sh)]

    def keep(self, idx):
        idx = list(idx)
        cached = self._solution_matches()
        omitted = np.ones(self.P, dtype=bool)
        omitted[idx] = False
        residual_unchanged = not np.any(self.beta[omitted])
        old_coverage = self._gram_rows[idx]
        self.A = self.eng.slice_cols(self.A, idx)
        self.U = self.U[:, idx]
        self.Aty = self.Aty[idx]
        self.norms = self.norms[idx]
        self.G = self.G[np.ix_(idx, idx)]
        # Cached columns cover a prefix of the old dictionary. Reordering a
        # retained subset may move an uncached entry into that prefix.
        idx_array = np.asarray(idx, dtype=np.intp)
        self._gram_rows = np.array([
            np.flatnonzero(idx_array >= coverage)[0]
            if np.any(idx_array >= coverage) else len(idx)
            for coverage in old_coverage
        ], dtype=np.intp)
        self.known = self._gram_rows == len(idx)
        old2new = {j: i for i, j in enumerate(idx)}
        self.S = [old2new[j] for j in self.S if j in old2new]
        self.beta = self.beta[idx]
        if cached and residual_unchanged:
            self._cached_beta = self.beta.copy()
            self._cached_corr = self._cached_corr[idx]
        else:
            self._invalidate_solution()

    def _ensure_gram(self, cols):
        # An append invalidates only new rows of existing Gram columns. Keep
        # their old prefix instead of repeating its full-height products.
        cols = np.asarray(cols, dtype=np.intp)
        for start in np.unique(self._gram_rows[cols]):
            if start >= self.P:
                continue
            need = cols[self._gram_rows[cols] == start]
            # Complete columns already supply their symmetric entries, notably
            # when all freshly appended columns were admitted together.
            rows = np.arange(start, self.P)
            rows = rows[self._gram_rows[rows] != self.P]
            if len(rows):
                row_idx = None if len(rows) == self.P else rows
                Gn = self.eng.cross_gram(self.A, need, row_idx=row_idx)
                self.G[np.ix_(rows, need)] = Gn
                self.G[np.ix_(need, rows)] = Gn.T
            self._gram_rows[need] = self.P
            self.known[need] = True

    @staticmethod
    def _kkt(beta, corr, lam):
        """Maximum absolute first-order residual in coefficient coordinates."""
        violation = np.maximum(np.abs(corr) - lam, 0.0)
        active = beta != 0.0
        violation[active] = np.abs(corr[active] - lam * np.sign(beta[active]))
        return violation

    @staticmethod
    def _orthant_step(G, h, penalty, z):
        """Accelerate coordinate descent without jumping across a sign face.

        A singular support is allowed: least squares supplies a direction, and
        the descent check rejects numerical directions that do not improve the
        original objective. No artificial ridge changes the optimum.
        """
        active = np.flatnonzero(z)
        if not len(active):
            return z
        Gs = G[np.ix_(active, active)]
        gradient = h[active] - G[active] @ z
        rhs = gradient - penalty[active] * np.sign(z[active])
        try:
            L = np.linalg.cholesky(Gs)
            # A tiny Cholesky pivot indicates a rank-deficient support at the
            # precision of the Gram matrix; use a rank-revealing solve instead.
            if np.min(np.diag(L)) <= 1e-7:
                raise np.linalg.LinAlgError
            if _solve_triangular is None:
                step = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
            else:
                step = _solve_triangular(
                    L.T,
                    _solve_triangular(L, rhs, lower=True, check_finite=False),
                    check_finite=False,
                )
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(Gs, rhs, rcond=None)[0]
            # With dependent features, the orthant objective may still have a
            # linear descent direction in the Gram null space. A pseudoinverse
            # step alone misses it and can stall far from a Lasso optimum.
            null_descent = rhs - Gs @ step
            if np.linalg.norm(null_descent) > 1e-12 * max(1.0, np.linalg.norm(rhs)):
                shrinking = z[active] * null_descent < 0.0
                if np.any(shrinking):
                    boundary = np.min(-z[active][shrinking] / null_descent[shrinking])
                    step = boundary * null_descent
        if not np.all(np.isfinite(step)) or float(rhs @ step) <= 0.0:
            return z
        candidate = z[active] + step
        crossing = np.sign(candidate) != np.sign(z[active])
        alpha = 1.0
        hit = np.zeros(len(active), dtype=bool)
        if np.any(crossing):
            ratios = np.full(len(active), np.inf)
            ratios[crossing] = -z[active][crossing] / step[crossing]
            alpha = min(1.0, float(np.min(ratios)))
            hit = ratios <= alpha * (1.0 + 8.0 * np.finfo(float).eps)
        updated = z.copy()
        updated[active] += alpha * step
        updated[active[hit]] = 0.0
        delta = updated - z
        # Evaluate the objective difference directly to avoid cancellation of
        # the common, possibly large y.T@y constant.
        change = float(
            -(h - G @ z) @ delta
            + 0.5 * delta @ (G @ delta)
            + penalty @ (np.abs(updated) - np.abs(z))
        )
        return updated if change <= 0.0 else z

    def _solve_working_set(self, indices, beta, lam, threshold, corr):
        self._ensure_gram(indices)
        norms = self.norms[indices]
        G = self.G[np.ix_(indices, indices)] / np.outer(norms, norms)
        penalty = lam / norms
        z = beta[indices] * norms
        # Anchor the small quadratic to the latest residual correlations. This
        # avoids repeatedly subtracting large A.T@y and Gram@beta terms when
        # full-data verification requests a small numerical correction.
        h = G @ z + corr[indices] / norms
        diagonal = np.diag(G)
        # Several cheap Gram sweeps can run between full-data screening passes.
        # Newton acceleration makes correlated active features converge without
        # waiting for slow coordinate-descent contraction along narrow valleys.
        for _ in range(100):
            gradient = h - G @ z
            for j in range(len(indices)):
                if diagonal[j] <= 0.0:
                    z[j] = 0.0
                    continue
                value = z[j] + gradient[j] / diagonal[j]
                updated = np.sign(value) * max(
                    abs(value) - penalty[j] / diagonal[j], 0.0
                )
                delta = updated - z[j]
                if delta:
                    z[j] = updated
                    gradient -= G[:, j] * delta
            z = self._orthant_step(G, h, penalty, z)
            corr = norms * (h - G @ z)
            if np.max(self._kkt(z, corr, lam), initial=0.0) <= threshold:
                break
        beta[indices] = z / norms

    def solve(self, lam, max_outer=None, tol=1e-8, batch_add=16):
        """Solve ``0.5 * ||A beta - y||² + lam * ||beta||_1``.

        Each working-set round admits up to ``batch_add`` violating columns.
        ``tol * max(1, lam)`` bounds the maximum absolute KKT residual, including
        stationarity on nonzero coefficients. ``max_outer`` limits working-set
        rounds; ``converged`` and ``stopping_reason`` distinguish a
        satisfied tolerance from an iteration limit. All diagnostics refer to
        the engine's stored features (which may be float32).

        Returns ``(beta, dual_residual_shards, objective, active_indices)``.
        """
        if not np.isfinite(lam) or lam < 0:
            raise ValueError("lam must be finite and nonnegative")
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be finite and positive")
        if not isinstance(batch_add, (int, np.integer)) or batch_add <= 0:
            raise ValueError("batch_add must be a positive integer")
        if max_outer is None:
            max_outer = 2 * self.P + 32
        if not isinstance(max_outer, (int, np.integer)) or max_outer < 0:
            raise ValueError("max_outer must be a nonnegative integer")
        threshold = tol * max(1.0, lam)
        beta = self.beta.copy()
        self.n_iter = 0
        self.converged = False
        self.stopping_reason = "iteration_limit"
        if self._solution_matches():
            residual = self._cached_residual
            corr = self._cached_corr.copy()
            rss = self._cached_rss
            verified = self._cached_verified
        elif not np.any(beta):
            residual = [-y for y in self.eng.ys]
            corr = self.Aty.copy()
            rss = None
            verified = True
        else:
            residual, negative_corr = self.eng.residual_and_screen(self.A, beta)
            corr = -negative_corr
            rss = None
            verified = True
        while True:
            violation = self._kkt(beta, corr, lam)
            self.kkt_violation = float(np.max(violation, initial=0.0))
            if self.kkt_violation <= threshold or self.n_iter >= max_outer:
                if not verified:
                    residual, negative_corr = self.eng.residual_and_screen(self.A, beta)
                    corr = -negative_corr
                    rss = None
                    verified = True
                    violation = self._kkt(beta, corr, lam)
                    self.kkt_violation = float(np.max(violation, initial=0.0))
                if self.kkt_violation <= threshold:
                    self.converged = True
                    self.stopping_reason = "kkt_tolerance"
                    break
                if self.n_iter >= max_outer:
                    break
            active = np.flatnonzero(beta)
            candidates = np.flatnonzero((beta == 0.0) & (violation > threshold))
            if len(candidates) > batch_add:
                # Partial selection avoids sorting the entire atom dictionary.
                chosen = np.argpartition(violation[candidates], -batch_add)[-batch_add:]
                candidates = candidates[chosen]
            candidates = candidates[np.argsort(-violation[candidates], kind="stable")]
            indices = np.concatenate([active, candidates])
            old_beta = beta[indices].copy()
            self._solve_working_set(indices, beta, lam, threshold, corr)
            delta = beta[indices] - old_beta
            changed = delta != 0.0
            corr -= self.G[:, indices[changed]] @ delta[changed]
            verified = False
            self.n_iter += 1
        self.beta = beta.copy()
        self.S = np.flatnonzero(beta).tolist()
        if rss is None:
            rss = sum(float(self.eng._host64((v * v).sum())) for v in residual)
        self._cached_beta = beta.copy()
        self._cached_residual = residual
        self._cached_corr = corr.copy()
        self._cached_rss = rss
        self._cached_verified = True
        objective = 0.5 * rss + lam * float(np.abs(beta).sum())
        return beta, [-v for v in residual], objective, list(self.S)
