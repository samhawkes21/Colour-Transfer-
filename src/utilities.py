"""
Optimal Transport Utilities.

This module implements core routines for entropic optimal transport
and nearest-neighbor style computations on high-dimensional point sets.

Functions
---------
cost(A, B, chunk)
    Computes the squared Euclidean cost matrix between two feature sets A and B
    using chunked computation to reduce memory usage. Returns a dense (n x m)
    float64 matrix.

sinkhorn_log(a, b, C, eps, max_iters, tol, check_every)
    Solves the entropic optimal transport problem using the Sinkhorn-Knopp
    algorithm in the log domain for numerical stability. Returns the optimal
    transport plan, the final convergence error, and number of iterations.
"""

from __future__ import annotations
import numpy as np
from scipy.special import logsumexp

def cost(A: np.ndarray, B: np.ndarray, chunk: int) -> np.ndarray:
    """
    Finds cost (squared euclidean) between A and B.
    Uses chunking over A to control memory.
    Returns C shape (n,m), float64.
    """
    n, d = A.shape
    m, _ = B.shape
    C = np.empty((n, m), dtype=np.float64)
    B2 = np.sum(B * B, axis=1, keepdims=True).T  
    print(f"[1/4] Building cost matrix C (shape {n}x{m})")

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        A_chunk = A[start:end]
        A2 = np.sum(A_chunk * A_chunk, axis=1, keepdims=True)  
        #||a-b||^2=||a||^2+||b||^2-2a^Tb
        C_chunk = A2 + B2 - 2.0 * (A_chunk @ B.T)
        C[start:end] = np.maximum(C_chunk, 0.0)

    return C

def sinkhorn_log(a: np.ndarray, b: np.ndarray, C: np.ndarray, eps: float,
                      max_iters: int, tol: float, check_every: int) -> tuple[np.ndarray, float, int]:
    """
    Entropic OT via log-domain Sinkhorn.
    Returns (P, err, iters).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)

    n, m = C.shape
    loga = np.log(a)
    logb = np.log(b)
    logK = -C / float(eps)
    logu = np.ones(n, dtype=np.float64)
    logv = np.ones(m, dtype=np.float64)

    err = np.inf
    it = 0

    for it in range(1, max_iters + 1):
        logKv = logsumexp(logK + logv[None, :], axis=1)
        logu = loga - logKv
        logKTu = logsumexp(logK + logu[:, None], axis=0)
        logv = logb - logKTu

        if it % check_every == 0 or it == max_iters:
            logKv = logsumexp(logK + logv[None, :], axis=1)
            logKTu = logsumexp(logK + logu[:, None], axis=0)

            r = np.exp(logu + logKv)
            c = np.exp(logv + logKTu)

            err_a = np.sum(np.abs(r - a))
            err_b = np.sum(np.abs(c - b))
            err = float(max(err_a, err_b))

            if err < tol:
                break

    P = np.exp(logu[:, None] + logK + logv[None, :])
    return P, err, it
