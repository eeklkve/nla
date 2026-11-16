import numpy as np
import pytest
from nla.schur import schur_by_induction


def check_schur(A, U, T):
    n = A.shape[0]
    assert np.allclose(U.conj().T @ U, np.eye(n), atol=1e-8)
    assert np.allclose(T, np.triu(T), atol=1e-8)
    assert np.allclose(U @ T @ U.conj().T, A, atol=1e-8)


def sort_key(z):
    return np.lexsort((z.imag, z.real))


@pytest.mark.parametrize("n", [1, 2, 5, 12])
def test_random_complex(n):
    rng = np.random.default_rng(n)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    U, T = schur_by_induction(A)
    check_schur(A, U, T)
    lam = np.linalg.eigvals(A)
    d = np.diag(T)
    assert np.allclose(d[sort_key(d)], lam[sort_key(lam)], atol=1e-6)


def test_real_matrix_with_complex_eigenvalues():
    t = 0.7
    A = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    U, T = schur_by_induction(A)
    check_schur(A, U, T)
    d = np.diag(T)
    assert np.allclose(np.sort(d.imag), [-np.sin(t), np.sin(t)])


def test_jordan_block():
    A = np.array([[2.0, 1.0], [0.0, 2.0]])
    U, T = schur_by_induction(A)
    check_schur(A, U, T)
    assert np.allclose(np.diag(T), [2.0, 2.0], atol=1e-6)


def test_hermitian_gives_diagonal_T():
    rng = np.random.default_rng(10)
    B = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
    A = B + B.conj().T
    U, T = schur_by_induction(A)
    check_schur(A, U, T)
    assert np.allclose(T, np.diag(np.diag(T)), atol=1e-8)
    assert np.allclose(np.diag(T).imag, 0, atol=1e-8)


def test_non_normal_has_nonzero_offdiagonal():
    A = np.array([[1.0, 5.0], [0.0, 2.0]])
    U, T = schur_by_induction(A)
    check_schur(A, U, T)
    assert abs(T[0, 1]) > 1.0
