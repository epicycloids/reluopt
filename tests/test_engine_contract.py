"""Input validation and chunk-bounded Gram reductions."""
import numpy as np
import pytest
from reluopt.data_engine import ShardedData


@pytest.mark.parametrize("X,y", [
    (np.zeros(3), np.zeros(3)), (np.empty((0, 2)), np.empty(0)),
    (np.empty((3, 0)), np.zeros(3)), (np.zeros((3, 2)), np.zeros((3, 1))),
    (np.full((3, 2), np.nan), np.zeros(3)),
    (np.zeros((3, 2)), np.array([0, np.inf, 1])),
    (np.array([[1j]]), np.zeros(1)),
])
def test_reject_invalid_data(X, y):
    with pytest.raises(ValueError):
        ShardedData(X, y)


@pytest.mark.parametrize("kwargs", [dict(dtype="float16"), dict(chunk=0),
    dict(chunk=1.5), dict(devices=[]), dict(weights=[-1]), dict(weights=[0]),
    dict(weights=[np.nan]), dict(weights=[1, 1])])
def test_reject_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        ShardedData(np.eye(3), np.ones(3), **kwargs)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gram_does_not_materialize_full_height_subset(dtype):
    rng = np.random.default_rng(4)
    eng = ShardedData(rng.normal(size=(23, 3)), rng.normal(size=23),
                      dtype=dtype, chunk=4, devices=["cpu"] * 2)
    A = eng.build_atoms(rng.normal(size=(3, 5)))
    host = np.concatenate(A).astype(np.float64)
    def forbidden(*args):
        raise AssertionError("full-height subset copy")
    eng.slice_cols = forbidden
    np.testing.assert_allclose(eng.gram(A, [1, 3]), host[:, [1, 3]].T @ host[:, [1, 3]], atol=1e-12)
    np.testing.assert_allclose(eng.cross_gram(A, [1, 3]), host.T @ host[:, [1, 3]], atol=1e-12)


def test_empty_column_slice():
    eng = ShardedData(np.eye(3), np.ones(3))
    A = eng.build_atoms(np.eye(3))
    assert eng.slice_cols(A, [])[0].shape == (3, 0)


def test_reject_complex_targets():
    with pytest.raises(ValueError, match="real"):
        ShardedData(np.ones((2, 1)), np.array([1+2j, 3+4j]))


def test_search_conversion_rejects_overflow():
    from reluopt.kernels import as_float32
    with pytest.raises(ValueError, match="rescale"):
        as_float32(np.array([1e40]))


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_fused_residual_and_selective_gram(dtype):
    rng=np.random.default_rng(82)
    eng=ShardedData(rng.normal(size=(31,4)),rng.normal(size=31),
                    dtype=dtype,chunk=5,devices=['cpu','cpu'])
    A=eng.build_atoms(rng.normal(size=(4,7)))
    beta=rng.normal(size=7)
    residual,corr=eng.residual_and_screen(A,beta)
    old=eng.residual(A,beta)
    np.testing.assert_allclose(eng.gather_vec(residual),eng.gather_vec(old),atol=1e-12)
    np.testing.assert_allclose(corr,eng.screen(A,old),atol=1e-12)
    host=np.concatenate(A).astype(np.float64)
    np.testing.assert_allclose(eng.cross_gram(A,[0,4],row_idx=[2,3,6]),
                                host[:,[2,3,6]].T@host[:,[0,4]],atol=1e-12)
