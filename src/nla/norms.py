import numpy as np


def vec_norm(x, p, axis=None):
    a = np.abs(x)
    if np.isinf(p):
        r = a.max(axis=axis)
    else:
        r = (a ** p).sum(axis=axis) ** (1 / p)
    return float(r) if axis is None else r


def frobenius(A):
    return np.sqrt((np.abs(A) ** 2).sum())


def op_norm_1(A):
    return np.sum(np.abs(A), axis=0).max()


def op_norm_inf(A):
    return np.sum(np.abs(A), axis=1).max()


def spectral_norm(A):
    eigs = np.linalg.eigvalsh(A.conj().T @ A)
    return (eigs.max()) ** (0.5)


def op_norm_lower_bound(A, p, n_samples=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((A.shape[1], n_samples))
    ratios = vec_norm(A @ x, p, axis=0) / vec_norm(x, p, axis=0)
    return float(ratios.max())

def random_unitary(n, seed=0):
    z = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    Q, R = np.linalg.qr(z)
    return Q * (np.diag(R) / np.abs(np.diag(R)))
