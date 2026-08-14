"""Downsampled, web-friendly layer exports for the Dash UI: one base RGB
image plus one transparent PNG per mask class, all the same pixel size so
they can be stacked with CSS. Masks are dilated before downsampling so thin
blobs (a lot of the young-tree candidates are just a few pixels wide) don't
disappear at preview resolution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, zoom

MATURE_COLOR = (220, 40, 40, 210)
YOUNG_COLOR = (0, 220, 220, 220)
OTHER_COLOR = (240, 150, 0, 190)


def _downsample_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1:
        return mask
    dilate_r = max(1, int(round(1 / scale)))
    dilated = binary_dilation(mask, iterations=dilate_r)
    return zoom(dilated.astype(np.uint8), scale, order=0) > 0


def _colored_overlay(mask_small: np.ndarray, rgba) -> Image.Image:
    h, w = mask_small.shape
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[mask_small] = rgba
    return Image.fromarray(arr, mode="RGBA")


def export_web_layers(outdir: Path, stack, mature_mask, other_veg_mask, young_mask, max_dim=1600):
    h, w = mature_mask.shape
    scale = min(1.0, max_dim / max(h, w))

    rgb = np.moveaxis(stack["rgb"], 0, -1)
    base_img = Image.fromarray(rgb)
    if scale < 1:
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        base_img = base_img.resize(new_size, Image.BILINEAR)
    base_img.save(outdir / "web_base_rgb.png")

    for name, mask, color in [
        ("web_mature_overlay.png", mature_mask, MATURE_COLOR),
        ("web_young_overlay.png", young_mask, YOUNG_COLOR),
        ("web_other_overlay.png", other_veg_mask, OTHER_COLOR),
    ]:
        small = _downsample_mask(mask, scale)
        _colored_overlay(small, color).save(outdir / name)

    return {"width": base_img.width, "height": base_img.height}
