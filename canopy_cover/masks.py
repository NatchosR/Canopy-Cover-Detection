"""Canopy mask construction: mature-tree detection (height-driven) and
young-tree *candidate* detection (spectral + planting-row-driven).

Findings from EDA (see scripts/eda.py, run on the real plot_1/plot_2 rasters)
that motivate this split:

- CHM noise floor over bare ground is very tight (std ~a few cm). Mature trees
  show up as a clean, separate bump in the CHM histogram above a valley around
  0.4-0.6 m -> height + ExG is a reliable detector for mature canopy.
- Young trees in plot_1 are visible by eye as small green tufts sitting right
  on the planting rows, but many produce NO usable CHM signal (below the
  DSM/DTM noise floor or smoothed out by photogrammetric interpolation) --
  median reported height from the tree-tracking file is ~43 cm. So mature-tree
  logic (height threshold) cannot reliably find them; detection falls back to
  ExG + proximity to known planting rows, and blobs are reported as
  "potential young trees" rather than confirmed counts.
- Cross-checked against data/plot_1/plot1_tree_locations.gpkg (6326 tracked
  trees): the blanket ExG/row-corridor candidate mask recovers roughly half
  of them (see canopy_cover.validate) -- a reasonable recall for a
  spectral-only detector on trees this small, not a precision failure.
"""
from __future__ import annotations

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, remove_small_holes, closing, disk


def pixel_area_m2(transform) -> float:
    return abs(transform.a * transform.e - transform.b * transform.d)


def clean_mask(mask: np.ndarray, min_area_px: int, closing_radius_px: int = 1) -> np.ndarray:
    """Remove speckle, close small gaps."""
    m = mask.copy()
    if closing_radius_px > 0:
        m = closing(m, footprint=disk(closing_radius_px))
    m = remove_small_objects(m, max_size=min_area_px)
    m = remove_small_holes(m, max_size=min_area_px)
    return m


def build_mature_tree_mask(chm, exg, valid, transform, height_thresh, exg_thresh, min_blob_m2):
    """Height + greenness driven mask for mature/clearly-identifiable trees."""
    px_area = pixel_area_m2(transform)
    min_area_px = max(1, int(round(min_blob_m2 / px_area)))
    raw = valid & (chm > height_thresh) & (exg > exg_thresh)
    return clean_mask(raw, min_area_px)


def split_existing_by_shape(mask, transform, circularity_thresh=0.3, min_area_for_shape_check_m2=3.0):
    """Split a tall-canopy mask into compact tree-like blobs vs. sprawling
    hedgerow/scrub/rock-bramble blobs, using shape only (not position).

    Rationale (from plot_2 inspection): actual orchard tree crowns are
    reasonably round even when large (circularity > ~0.3). Boundary hedges
    and natural scrub clusters are elongated/irregular (circularity < 0.3)
    and are usually -- but not always -- larger than a single tree crown, so
    the area gate avoids reclassifying small, noisily-shaped tree blobs.
    Blobs below `min_area_for_shape_check_m2` are always kept as trees since
    circularity is unstable for a handful of pixels.

    Returns (tree_mask, other_veg_mask, blob_stats).
    """
    px_area = pixel_area_m2(transform)
    lbl = label(mask, connectivity=2)
    tree_mask = np.zeros_like(mask)
    other_mask = np.zeros_like(mask)
    n_other = 0
    for prop in regionprops(lbl):
        area_m2 = prop.area * px_area
        circularity = 4 * np.pi * prop.area / (prop.perimeter ** 2) if prop.perimeter > 0 else 1.0
        is_other = area_m2 > min_area_for_shape_check_m2 and circularity < circularity_thresh
        if is_other:
            other_mask[lbl == prop.label] = True
            n_other += 1
        else:
            tree_mask[lbl == prop.label] = True
    return tree_mask, other_mask, {"n_other_vegetation_blobs": n_other}


def build_young_tree_candidate_mask(exg, valid, transform, exg_thresh, min_blob_m2,
                                     compact_size_m2, circularity_thresh,
                                     row_proximity_mask=None, exclude_mask=None):
    """Spectral *candidate* mask for young/small trees not yet tall enough to
    clear the CHM noise floor (or not clearly identifiable as mature). Blobs
    are "potential young trees", not confirmed detections -- see
    canopy_cover.validate for a recall check against tracked tree locations
    where available.

    Blobs smaller than `min_blob_m2` are dropped as noise. Blobs up to
    `compact_size_m2` are kept unconditionally (shape is unstable to estimate
    for a handful of pixels, and this covers genuinely tiny sapling tufts).
    Larger blobs are only kept if they're reasonably round (circularity >=
    `circularity_thresh`) -- this is what actually separates a real tree
    crown (round, of any size -- crowns range from <0.1 m^2 fresh sprouts to
    >10 m^2 well-grown young trees depending on species/age) from a sprawling
    grass/weed mat that happens to pass the ExG threshold. A flat size cap
    doesn't work across plots: it wrongly rejects legitimate large round
    crowns (see plot_3 young almonds, up to ~12 m^2).

    row_proximity_mask: boolean array, True within a buffer of known planting
    rows. If None, no spatial prior is available (e.g. plot_2, plot_3) and
    detection runs on ExG + blob shape/size alone, which is expected to be
    less precise.
    exclude_mask: pixels already claimed by the mature-tree mask are excluded
    to avoid double counting.
    """
    px_area = pixel_area_m2(transform)
    min_area_px = max(1, int(round(min_blob_m2 / px_area)))
    compact_area_px = int(round(compact_size_m2 / px_area))

    raw = valid & (exg > exg_thresh)
    if row_proximity_mask is not None:
        raw = raw & row_proximity_mask
    if exclude_mask is not None:
        raw = raw & ~exclude_mask

    raw = clean_mask(raw, min_area_px, closing_radius_px=0)

    lbl = label(raw, connectivity=2)
    keep = np.zeros_like(raw)
    n_blobs_before = 0
    n_blobs_kept = 0
    n_rejected_small = 0
    n_rejected_shape = 0
    for prop in regionprops(lbl):
        n_blobs_before += 1
        if prop.area < min_area_px:
            n_rejected_small += 1
            continue
        if prop.area > compact_area_px:
            circularity = 4 * np.pi * prop.area / (prop.perimeter ** 2) if prop.perimeter > 0 else 1.0
            if circularity < circularity_thresh:
                n_rejected_shape += 1
                continue
        keep[lbl == prop.label] = True
        n_blobs_kept += 1

    return keep, {
        "n_blobs_before_filter": n_blobs_before, "n_blobs_kept": n_blobs_kept,
        "n_rejected_too_small": n_rejected_small, "n_rejected_sprawling_shape": n_rejected_shape,
    }
