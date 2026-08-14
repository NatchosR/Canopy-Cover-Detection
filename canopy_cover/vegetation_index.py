"""RGB-only vegetation indices (no NIR available)."""
import numpy as np


def excess_green(rgb_uint8: np.ndarray) -> np.ndarray:
    """ExG = 2G - R - B on a (3,H,W) or (H,W,3) uint8 array, RGB normalized to [0,1]."""
    if rgb_uint8.shape[0] == 3:
        r, g, b = rgb_uint8[0], rgb_uint8[1], rgb_uint8[2]
    else:
        r, g, b = rgb_uint8[..., 0], rgb_uint8[..., 1], rgb_uint8[..., 2]
    r = r.astype(np.float32) / 255.0
    g = g.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0
    return 2 * g - r - b


def vari(rgb_uint8: np.ndarray) -> np.ndarray:
    """VARI = (G-R) / (G+R-B), on a (3,H,W) or (H,W,3) uint8 array."""
    if rgb_uint8.shape[0] == 3:
        r, g, b = rgb_uint8[0], rgb_uint8[1], rgb_uint8[2]
    else:
        r, g, b = rgb_uint8[..., 0], rgb_uint8[..., 1], rgb_uint8[..., 2]
    r = r.astype(np.float32) / 255.0
    g = g.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0
    denom = g + r - b
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(denom) > 1e-6, (g - r) / denom, 0.0)
    return out.astype(np.float32)
