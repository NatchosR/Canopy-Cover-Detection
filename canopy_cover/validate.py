"""Validation against a tracked/known tree location file.

The plot_1 tree-location file has the correct tree sequence but inaccurate
coordinates (and partial, also-inaccurate height records) -- it is a spatial
prior, not field-verified ground truth. What it *is* good for: a coarse
recall check -- "for each tracked tree, is there a detected blob nearby?" --
which is exactly the kind of check CLAUDE.md asks for before trusting a
canopy KPI. This is not precision/recall in the strict sense (no confirmed
negatives), just a recall proxy plus an unmatched-blob count.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import label, regionprops

from .masks import pixel_area_m2


def _blob_centroids_m(mask, transform):
    px_area = pixel_area_m2(transform)
    lbl = label(mask, connectivity=2)
    centroids = []
    for prop in regionprops(lbl):
        y, x = prop.centroid
        mx, my = transform * (x, y)
        centroids.append((mx, my))
    return np.array(centroids) if centroids else np.zeros((0, 2))


def _greedy_one_to_one_match(known_xy, cand_xy, cand_kind, max_radius_m):
    """Greedy nearest-first 1:1 assignment between known points and candidate
    blob centroids (which may mix young + mature blobs). Necessary because
    average tree spacing here (~1.8 m) is smaller than a generous match
    radius -- a naive nearest-neighbour-per-point check lets one blob
    "explain" several nearby known trees and badly overstates recall.
    """
    n_known = len(known_xy)
    if n_known == 0 or len(cand_xy) == 0:
        return np.zeros(n_known, dtype=bool), np.array([], dtype=object), np.zeros(len(cand_xy), dtype=bool)

    known_tree = cKDTree(known_xy)
    pairs = known_tree.query_ball_point(cand_xy, r=max_radius_m)

    dist_pairs = []
    for cand_idx, known_idxs in enumerate(pairs):
        for known_idx in known_idxs:
            d = np.linalg.norm(cand_xy[cand_idx] - known_xy[known_idx])
            dist_pairs.append((d, known_idx, cand_idx))
    dist_pairs.sort(key=lambda t: t[0])

    known_used = np.zeros(n_known, dtype=bool)
    cand_used = np.zeros(len(cand_xy), dtype=bool)
    matched_kind = np.full(n_known, None, dtype=object)

    for d, known_idx, cand_idx in dist_pairs:
        if known_used[known_idx] or cand_used[cand_idx]:
            continue
        known_used[known_idx] = True
        cand_used[cand_idx] = True
        matched_kind[known_idx] = cand_kind[cand_idx]

    return known_used, matched_kind, cand_used


def validate_against_tree_locations(young_mask, mature_mask, transform, points_gdf, match_radius_m=1.5):
    """For each known tree point, check whether a detected blob (young or
    mature) can be matched to it 1:1 within `match_radius_m`. Returns
    recall-style stats. Matching is exclusive (greedy nearest-first) so a
    single blob cannot inflate recall by "covering" several nearby known
    trees -- tree spacing here (~1.8 m) is comparable to a generous match
    radius, so this matters a lot.
    """
    known_xy = np.array([(geom.x, geom.y) for geom in points_gdf.geometry])
    young_xy = _blob_centroids_m(young_mask, transform)
    mature_xy = _blob_centroids_m(mature_mask, transform)

    cand_xy = np.vstack([young_xy, mature_xy]) if len(young_xy) or len(mature_xy) else np.zeros((0, 2))
    cand_kind = np.array(["young"] * len(young_xy) + ["mature"] * len(mature_xy), dtype=object)

    known_used, matched_kind, cand_used = _greedy_one_to_one_match(known_xy, cand_xy, cand_kind, match_radius_m)
    n_known = len(known_xy)

    n_matched_young = int((matched_kind == "young").sum())
    n_matched_mature = int((matched_kind == "mature").sum())
    n_young_blobs_unmatched = int(len(young_xy) - cand_used[:len(young_xy)].sum()) if len(young_xy) else 0

    return {
        "match_radius_m": match_radius_m,
        "matching": "greedy nearest-first, exclusive 1:1 (a blob can match at most one known tree)",
        "n_known_trees": int(n_known),
        "n_known_matched_by_young_blob": n_matched_young,
        "n_known_matched_by_mature_blob": n_matched_mature,
        "n_known_matched_total": int(known_used.sum()),
        "n_known_unmatched": int(n_known - known_used.sum()),
        "recall_pct": float(100 * known_used.sum() / n_known) if n_known else None,
        "n_young_blobs_total": int(len(young_xy)),
        "n_young_blobs_unmatched": n_young_blobs_unmatched,
        "note": "Recall proxy only: tree-location file has correct sequence but "
                "inaccurate coordinates, not field-verified ground truth. Not a "
                "substitute for precision validation against confirmed reference points.",
    }
