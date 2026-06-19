"""
Image Feature Extraction and Nearest-Neighbor Matching Utilities.

This module provides functions for constructing per-pixel feature representations
of RGB images and performing efficient nearest-neighbor matching between two
feature sets using squared Euclidean distance.

Functions
---------
make_features(img_255, w_pos)
    Converts an RGB image into a feature matrix where each pixel is represented
    by its color (R, G, B) and scaled spatial coordinates (x, y). The spatial
    weighting parameter `w_pos` controls the influence of positional information.

nearest_neighbor_map(A, B, chunk)
    Computes a nearest-neighbor assignment from each row in A to the closest
    row in B under squared Euclidean distance. Computation is chunked over A
    for memory efficiency on large datasets.
"""

from __future__ import annotations
import numpy as np
from scipy.special import logsumexp

def make_features(img_255: np.ndarray, w_pos: float) -> np.ndarray:
    """
    Per-pixel feature vectors:
    [R,G,B, w_pos*px, w_pos*py] where px,py in [0,255] (global coords)
    Returns float64 array (N,d).
    """
    H, W, _ = img_255.shape
    rgb = img_255.reshape(-1, 3).astype(np.float64)

    ys = np.repeat(np.arange(H), W).astype(np.float64)  
    xs = np.tile(np.arange(W), H).astype(np.float64)    

    px = np.zeros_like(xs) if W == 1 else 255.0 * xs / (W - 1)
    py = np.zeros_like(ys) if H == 1 else 255.0 * ys / (H - 1)

    pos = np.stack([px, py], axis=1)  
    return np.concatenate([rgb, w_pos * pos], axis=1)  

def nearest_neighbor_map(A: np.ndarray, B: np.ndarray, chunk: int) -> np.ndarray:
    """
    For each pixel in A, finds nearest pixel in B by squared Euclidean distance.
    Returns idx shape (N,), int.
    Chunked over A with progress.
    """
    N, d = A.shape
    M, _ = B.shape
    idx = np.empty(N, dtype=np.int64)

    B2 = np.sum(B * B, axis=1)  # (M,)

    print(f"[4/4] Nearest-neighbor mapping: {N} points -> {M} samples")

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        A_chunk = A[start:end]                  
        A2 = np.sum(A_chunk * A_chunk, axis=1)    
        D = A2[:, None] + B2[None, :] - 2.0 * (A_chunk @ B.T)
        idx[start:end] = np.argmin(D, axis=1)

    return idx
