"""Data-pass and warm-state regressions for the finite master solver."""

import numpy as np

from reluopt.restricted_master import AtomLasso
from reluopt.data_engine import ShardedData


def _make_master(seed=37, rows=30, columns=8):
    rng = np.random.default_rng(seed)
    A = np.maximum(rng.standard_normal((rows, columns)), 0)
    y = rng.standard_normal(rows)
    master = AtomLasso(ShardedData(A, y, chunk=7))
    master.set_atoms(np.eye(columns))
    return master, A, y


def _count_passes(monkeypatch, engine):
    calls = {"screen": 0, "residual_and_screen": 0, "cross_gram": 0}
    for name in calls:
        original = getattr(engine, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(engine, name, counted)
    return calls


def _check(master, A, y, lam):
    beta, shards, objective, active = master.solve(lam, batch_add=1)
    residual = y - A @ beta
    corr = A.T @ residual
    np.testing.assert_allclose(master.eng.gather_vec(shards), residual, atol=1e-12)
    np.testing.assert_allclose(
        objective, 0.5 * residual @ residual + lam * np.abs(beta).sum(), atol=1e-12
    )
    assert master.converged
    assert master.kkt_violation <= 1e-8 * max(1, lam)
    np.testing.assert_allclose(
        master.kkt_violation, master._kkt(beta, corr, lam).max(initial=0), atol=1e-12
    )
    assert active == np.flatnonzero(beta).tolist()
    return beta, shards, objective


def test_admission_rounds_use_one_full_data_verification(monkeypatch):
    master, A, y = _make_master()
    calls = _count_passes(monkeypatch, master.eng)
    _check(master, A, y, 0.1)
    assert master.n_iter >= 3
    assert calls["residual_and_screen"] == 1
    assert calls["screen"] == 0


def test_unchanged_and_zero_pruned_models_reuse_verified_state(monkeypatch):
    master, A, y = _make_master()
    beta, returned_dual, objective = _check(master, A, y, 1.0)
    # Returned arrays are independent of cached coefficients and residuals.
    beta[:] = 99
    returned_dual[0][:] = 99
    calls = _count_passes(monkeypatch, master.eng)
    _, _, repeated_objective = _check(master, A, y, 1.0)
    assert repeated_objective == objective
    keep = np.flatnonzero(master.beta)
    assert len(keep) < master.P
    master.keep(keep)
    _check(master, A[:, keep], y, 1.0)
    assert calls == {"screen": 0, "residual_and_screen": 0, "cross_gram": 0}
    assert master.n_iter == 0


def test_external_warm_start_and_nonzero_pruning_invalidate_cache(monkeypatch):
    master, A, y = _make_master()
    _check(master, A, y, 0.1)
    calls = _count_passes(monkeypatch, master.eng)
    master.beta *= 0.75
    _check(master, A, y, 0.1)
    assert calls["residual_and_screen"] >= 2
    keep = np.arange(master.P)[np.arange(master.P) != master.S[0]]
    master.keep(keep)
    before = calls["residual_and_screen"]
    _check(master, A[:, keep], y, 0.1)
    assert calls["residual_and_screen"] >= before + 1


def test_appended_gram_columns_only_fill_missing_rows(monkeypatch):
    rng = np.random.default_rng(41)
    A = np.maximum(rng.standard_normal((25, 6)), 0)
    engine = ShardedData(A, rng.standard_normal(25), chunk=7)
    master = AtomLasso(engine)
    master.set_atoms(np.eye(6)[:, :3])
    master._ensure_gram([0, 1])
    old = master.G.copy()
    master.append_atoms(np.eye(6)[:, 3:])
    requested_rows = []
    original = engine.cross_gram

    def counted(A_sh, idx, row_idx=None):
        requested_rows.append(row_idx)
        return original(A_sh, idx, row_idx=row_idx)

    monkeypatch.setattr(engine, "cross_gram", counted)
    master._ensure_gram([0, 1])
    assert len(requested_rows) == 1
    np.testing.assert_array_equal(requested_rows[0], [3, 4, 5])
    np.testing.assert_array_equal(master.G[:3, :3], old)
    np.testing.assert_allclose(master.G[:, :2], (A.T @ A)[:, :2], atol=1e-12)
    # A reordered keep remains valid even when it puts uncached rows first.
    keep = [5, 0, 3, 1]
    master.keep(keep)
    master._ensure_gram(range(4))
    np.testing.assert_allclose(master.G, A[:, keep].T @ A[:, keep], atol=1e-12)


def test_append_rescores_only_new_columns_and_preserves_fit(monkeypatch):
    rng = np.random.default_rng(57)
    A = np.maximum(rng.standard_normal((24, 6)), 0)
    y = rng.standard_normal(24)
    engine = ShardedData(A, y, chunk=5)
    master = AtomLasso(engine)
    master.set_atoms(np.eye(6)[:, :3])
    _check(master, A[:, :3], y, 0.2)
    calls = _count_passes(monkeypatch, engine)
    master.append_atoms(np.eye(6)[:, 3:])
    assert calls["screen"] == 2  # Anew.T@y and Anew.T@cached_residual.
    assert calls["residual_and_screen"] == 0
    for lam in (0.2, 0.7, 0.02):
        _, _, objective = _check(master, A, y, lam)
        fresh = AtomLasso(ShardedData(A, y, chunk=5))
        fresh.set_atoms(np.eye(6))
        _, _, expected = _check(fresh, A, y, lam)
        np.testing.assert_allclose(objective, expected, atol=1e-11)


def test_new_gram_columns_supply_symmetric_old_column_extensions(monkeypatch):
    rng = np.random.default_rng(73)
    A = np.maximum(rng.standard_normal((20, 6)), 0)
    engine = ShardedData(A, rng.standard_normal(20))
    master = AtomLasso(engine)
    master.set_atoms(np.eye(6)[:, :3])
    master._ensure_gram([0, 1])
    master.append_atoms(np.eye(6)[:, 3:])
    calls = _count_passes(monkeypatch, engine)
    master._ensure_gram([0, 1, 3, 4, 5])
    # Products for the three new columns already contain every missing entry
    # of the two previously cached columns; do not multiply them again.
    assert calls["cross_gram"] == 1
    np.testing.assert_allclose(master.G[:, [0, 1, 3, 4, 5]], (A.T @ A)[:, [0, 1, 3, 4, 5]])


def test_numpy_only_triangular_fallback(monkeypatch):
    import reluopt.restricted_master as module

    master, A, y = _make_master()
    _, _, accelerated_objective = _check(master, A, y, 0.1)
    monkeypatch.setattr(module, "_solve_triangular", None)
    fallback, _, _ = _make_master()
    _, _, fallback_objective = _check(fallback, A, y, 0.1)
    np.testing.assert_allclose(fallback_objective, accelerated_objective, atol=1e-12)


def test_reused_pricing_correlations_do_not_replace_final_kkt_check(monkeypatch):
    A = np.eye(3)
    y = np.array([1.0, 2.0, 3.0])
    engine = ShardedData(A, y)
    master = AtomLasso(engine)
    master.set_atoms(A[:, :1])
    _check(master, A[:, :1], y, 0.1)
    calls = _count_passes(monkeypatch, engine)
    master.append_atoms(
        A[:, 1:], A_sh=[A[:, 1:]], aty=y[1:], sqnorms=np.ones(2),
        # Deliberately stale pricing correlations would incorrectly hide both
        # new variables if they were accepted as the final KKT diagnostic.
        dual_corr=np.zeros(2),
    )
    assert calls["screen"] == 0
    beta, _, _ = _check(master, A, y, 0.1)
    np.testing.assert_allclose(beta, y - 0.1)
    assert calls["residual_and_screen"] >= 2
