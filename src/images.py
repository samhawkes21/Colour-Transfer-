"""
Image utilities for color transfer project.

Includes:
- image loading/saving
- visualization tools
"""

from __future__ import annotations
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_rgb_255(path: str) -> np.ndarray:
    """Load image as float32 RGB in [0,255], shape (H,W,3)"""
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.float32)

def save_rgb_255(path: str, img_255: np.ndarray) -> None:
    """Save float image to disk as uint8"""
    img = np.clip(img_255, 0, 255).astype(np.uint8)
    Image.fromarray(img, mode="RGB").save(path)

def display_triptych(src_255: np.ndarray, tgt_255: np.ndarray, out_255: np.ndarray,
                     titles=("Source", "Target", "Output")) -> None:
    """Display source/target/output"""

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, img, title in zip(axes, [src_255, tgt_255, out_255], titles):
        ax.imshow(np.clip(img, 0, 255).astype(np.uint8))
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    return