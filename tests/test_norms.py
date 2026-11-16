import numpy as np
import pytest
from nla.norms import (
    vec_norm,
    frobenius,
    op_norm_1,
    op_norm_inf,
    spectral_norm,
    op_norm_lower_bound,
    random_unitary,
)


def rand_complex(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


@pytest.mark.parametrize("p", [1, 1.5, 2, 3, np.inf])
def test_vec_norm_matches_numpy(p):
    x = rand_complex(50, 1)
    assert np.isclose(vec_norm(x, p), np.linalg.norm(x, p))


@pytest.mark.parametrize("p", [1, 2, 4, np.inf])
def test_vec_norm_axioms(p):
    x = rand_complex(30, 2)
    y = rand_complex(30, 3)
    c = 2.0 - 1.5j
    assert vec_norm(x + y, p) <= vec_norm(x, p) + vec_norm(y, p) + 1e-12
    assert np.isclose(vec_norm(c * x, p), abs(c) * vec_norm(x, p))
    assert vec_norm(np.zeros(30), p) == 0


def test_vec_norm_below_one_breaks_triangle():
    x = np.array([1.0, 0.0])
    y = np.array([0.0, 1.0])
    assert vec_norm(x + y, 0.5) > vec_norm(x, 0.5) + vec_norm(y, 0.5)


def test_matrix_norms_match_numpy():
    A = rand_complex((7, 5), 4)
    assert np.isclose(frobenius(A), np.linalg.norm(A, "fro"))
    assert np.isclose(op_norm_1(A), np.linalg.norm(A, 1))
    assert np.isclose(op_norm_inf(A), np.linalg.norm(A, np.inf))
    assert np.isclose(spectral_norm(A), np.linalg.norm(A, 2))


@pytest.mark.parametrize("p", [1, 2, np.inf])
def test_lower_bound_is_below_and_close(p):
    A = np.random.default_rng(5).standard_normal((4, 4))
    true = {1: op_norm_1, 2: spectral_norm, np.inf: op_norm_inf}[p](A)
    lb = op_norm_lower_bound(A, p, n_samples=5000, seed=1)
    assert lb <= true + 1e-12
    assert lb >= 0.7 * true


def test_norm_inequalities():
    rng = np.random.default_rng(6)
    for _ in range(20):
        A = rng.standard_normal((6, 4))
        s2 = spectral_norm(A)
        sf = frobenius(A)
        r = np.linalg.matrix_rank(A)
        assert s2 <= sf + 1e-12
        assert sf <= np.sqrt(r) * s2 + 1e-12
        assert s2 <= np.sqrt(op_norm_1(A) * op_norm_inf(A)) + 1e-12


def test_submultiplicativity():
    rng = np.random.default_rng(7)
    A = rng.standard_normal((5, 6))
    B = rng.standard_normal((6, 3))
    for f in (op_norm_1, op_norm_inf, spectral_norm, frobenius):
        assert f(A @ B) <= f(A) * f(B) + 1e-12


def test_random_unitary():
    Q = random_unitary(6, seed=3)
    assert np.allclose(Q.conj().T @ Q, np.eye(6))
    assert np.allclose(Q @ Q.conj().T, np.eye(6))
    assert np.iscomplexobj(Q)


def test_unitary_invariance():
    n = 6
    Q = random_unitary(n, seed=4)
    Z = random_unitary(n, seed=5)
    A = rand_complex((n, n), 8)
    x = rand_complex(n, 9)
    assert np.isclose(vec_norm(Q @ x, 2), vec_norm(x, 2))
    assert np.isclose(spectral_norm(Q @ A @ Z), spectral_norm(A))
    assert np.isclose(frobenius(Q @ A @ Z), frobenius(A))
    assert not np.isclose(op_norm_1(Q @ A), op_norm_1(A))
    assert not np.isclose(op_norm_inf(Q @ A), op_norm_inf(A))
