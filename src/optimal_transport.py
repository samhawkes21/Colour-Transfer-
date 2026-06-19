"""
Optimal Transport Color Transfer Pipeline.

This module implements a full end-to-end color transfer algorithm between
a source and target image using entropic optimal transport. The pipeline
operates in a joint feature space combining RGB values and spatial pixel
coordinates, enabling structurally-aware color mapping.

The method proceeds as follows:
1. Load source and target images.
2. Construct per-pixel feature representations (color + position).
3. Uniformly subsample pixels from both images for tractable OT computation.
4. Compute a squared Euclidean cost matrix between sampled features.
5. Solve the entropic optimal transport problem using log-domain Sinkhorn.
6. Compute barycentric color projections from target to source samples.
7. Propagate colors to all source pixels via nearest-neighbor assignment.
8. Reconstruct and save the color-transferred output image.
9. Display source, target, and output side-by-side.

Function
--------
colour_transfer(src_path, tgt_path, out_path, eps, sinkhorn_iters,
                sinkhorn_tol, check_every, w_pos, ns, nt, seed,
                cost_chunk, nn_chunk)
    Executes the full optimal transport color transfer pipeline and saves
    the resulting image to disk.
"""

from __future__ import annotations
from images import load_rgb_255
from images import save_rgb_255
from images import display_triptych
from pixels import make_features
from pixels import nearest_neighbor_map
from utilities import cost
from utilities import sinkhorn_log
import numpy as np
from scipy.special import logsumexp

def colour_transfer(src_path: str, tgt_path: str, out_path: str, 
                       eps: float, sinkhorn_iters: int, sinkhorn_tol: float, check_every: int, 
                       w_pos: float, 
                       ns: int, nt: int, seed: int, 
                       cost_chunk: int, nn_chunk: int) -> None:
    """
    Run global OT colour transfer and save output.
    """
    rng = np.random.default_rng(seed)

    src = load_rgb_255(src_path)
    tgt = load_rgb_255(tgt_path)

    Hs, Ws, _ = src.shape
    Ht, Wt, _ = tgt.shape
    print(f"Source: {Hs}x{Ws}, Target: {Ht}x{Wt}")

    Xs = make_features(src, w_pos)
    Xt = make_features(tgt, w_pos)

    Ns = Xs.shape[0]
    Nt = Xt.shape[0]
    ns_eff = min(ns, Ns)
    nt_eff = min(nt, Nt)
    print(f"[0/4] Sampling pixels for OT: ns={ns_eff}/{Ns}, nt={nt_eff}/{Nt} (seed={seed})")

    idx_s = rng.choice(Ns, size=ns_eff, replace=False)
    idx_t = rng.choice(Nt, size=nt_eff, replace=False)
    Xs_s = Xs[idx_s]  
    Xt_t = Xt[idx_t]  
    a = np.full(ns_eff, 1.0 / ns_eff, dtype=np.float64)
    b = np.full(nt_eff, 1.0 / nt_eff, dtype=np.float64)

    C = cost(Xs_s, Xt_t, cost_chunk)
    P, err, iters = sinkhorn_log(a, b, C, eps, sinkhorn_iters, sinkhorn_tol, check_every)

    print(f"[3/4] Barycentric projection to colours (Sinkhorn iters={iters}, final err={err:.3e})")

    tgt_rgb = tgt.reshape(-1, 3).astype(np.float64)
    Y = tgt_rgb[idx_t]  
    row_sums = np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
    barycentric_rgb_samples = (P @ Y) / row_sums  

    nn = nearest_neighbor_map(Xs, Xs_s, nn_chunk)  
    out_rgb = barycentric_rgb_samples[nn].reshape(Hs, Ws, 3).clip(0, 255).astype(np.float32)

    save_rgb_255(out_path, out_rgb)
    print(f"Done. Saved: {out_path}")
    display_triptych(src, tgt, out_rgb, titles=("Source", "Target", "Output"))