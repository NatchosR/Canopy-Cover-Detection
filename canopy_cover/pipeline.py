"""Canopy cover computation for an agroforestry plot.

KPI structure (per user decision, revised 2026-08-17):

- KPI1 total_canopy_pct  = KPI2 + KPI3 + other vegetation. Shrubs/hedges are
                           still canopy, so they count toward the total --
                           they just aren't broken out as their own numbered
                           KPI (see below).
- KPI2 mature_canopy_pct = trees clearly identifiable as trees (height +
                           greenness + compact shape) -- olives, oaks,
                           almonds, whatever species, doesn't matter, just
                           needs to be unambiguously a tree crown.
- KPI3 young_canopy_pct  = "potential young trees": spectral candidate blobs
                           too small/short to be unambiguous. Always reported
                           with both a % and a blob count, since the count is
                           itself informative (see canopy_cover.validate).
- other_vegetation_pct   = NOT its own numbered KPI, but included in KPI1.
                           Shrub/hedge/"mato" -- tracked and exported as its
                           own layer for transparency about what's driving
                           the total.

Call `compute_canopy_cover(...)` directly, or use `python -m canopy_cover.cli`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from skimage.measure import label, regionprops

from .raster_utils import load_aligned_stack, compute_chm
from .vegetation_index import excess_green
from .masks import (
    build_mature_tree_mask, build_young_tree_candidate_mask, split_existing_by_shape,
    pixel_area_m2,
)
from .validate import validate_against_tree_locations
from .web_export import export_web_layers


@dataclass
class CanopyConfig:
    # Mature-tree detection (height-driven; validated against CHM histogram
    # valley + visual check, see scripts/eda.py outputs).
    mature_height_thresh_m: float = 0.5
    mature_exg_thresh: float = 0.05
    mature_min_blob_m2: float = 0.05

    # Young-tree *candidate* detection (spectral, since young trees are often
    # below the CHM noise floor -- see module docstring in masks.py).
    young_exg_thresh: float = 0.10
    young_row_buffer_m: float = 0.4
    young_min_blob_m2: float = 0.01
    # Blobs above this size are only kept if they're reasonably round (see
    # build_young_tree_candidate_mask) -- rejects sprawling grass/weed mats
    # while still allowing large, well-grown, round young-tree crowns.
    young_compact_size_m2: float = 0.5
    young_circularity_thresh: float = 0.35

    # Recall check against a tracked (not necessarily accurate) tree-location file.
    tree_location_match_radius_m: float = 1.5

    # Shape-based split of the mature-tree mask into orchard trees vs. other
    # woody vegetation (hedgerows, boundary scrub, rock-bramble clusters).
    other_veg_circularity_thresh: float = 0.3
    other_veg_min_area_m2: float = 3.0


def _rasterize_boundary(boundary_path, layer, transform, shape):
    gdf = gpd.read_file(boundary_path, layer=layer) if layer else gpd.read_file(boundary_path)
    mask = rasterio.features.rasterize(
        [(geom, 1) for geom in gdf.geometry], out_shape=shape, transform=transform, fill=0, dtype=np.uint8,
    )
    return mask.astype(bool), gdf


def _rasterize_row_buffer(rows_path, layer, buffer_m, crs, transform, shape):
    gdf = gpd.read_file(rows_path, layer=layer) if layer else gpd.read_file(rows_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    buffered = gpd.GeoSeries(gdf.geometry, crs=crs).buffer(buffer_m)
    mask = rasterio.features.rasterize(
        [(geom, 1) for geom in buffered], out_shape=shape, transform=transform, fill=0, dtype=np.uint8,
    )
    return mask.astype(bool), len(gdf)


def compute_canopy_cover(
    orthophoto_path, dsm_path, dtm_path, boundary_path,
    boundary_layer=None,
    planting_rows_path=None, planting_rows_layer=None,
    tree_locations_path=None, tree_locations_layer=None,
    config: CanopyConfig | None = None,
    outdir: str | None = None,
    plot_name: str = "plot",
) -> dict:
    cfg = config or CanopyConfig()

    stack = load_aligned_stack(orthophoto_path, dsm_path, dtm_path)
    chm, chm_valid = compute_chm(stack["dsm"], stack["dtm"], stack["dsm_mask"], stack["dtm_mask"])
    exg = excess_green(stack["rgb"])

    boundary_mask, boundary_gdf = _rasterize_boundary(
        boundary_path, boundary_layer, stack["transform"], chm.shape
    )
    valid = chm_valid & boundary_mask

    mature_raw_mask = build_mature_tree_mask(
        chm, exg, valid, stack["transform"],
        cfg.mature_height_thresh_m, cfg.mature_exg_thresh, cfg.mature_min_blob_m2,
    )
    mature_mask, other_veg_mask, shape_meta = split_existing_by_shape(
        mature_raw_mask, stack["transform"],
        cfg.other_veg_circularity_thresh, cfg.other_veg_min_area_m2,
    )
    tall_mask = mature_mask | other_veg_mask  # avoid double-counting young candidates

    young_mask = np.zeros_like(mature_mask)
    young_meta = {"mode": "exg_only", "note": "No planting-row file supplied; detection runs on ExG + blob shape/size alone (no spatial prior)."}
    row_mask = None

    if planting_rows_path is not None:
        row_mask, n_rows = _rasterize_row_buffer(
            planting_rows_path, planting_rows_layer, cfg.young_row_buffer_m,
            stack["crs"], stack["transform"], chm.shape,
        )
        young_meta = {"mode": "planting_rows", "n_rows": n_rows,
                      "note": "Candidate detection restricted to a buffer around known planting rows."}

    young_mask, ym = build_young_tree_candidate_mask(
        exg, valid, stack["transform"], cfg.young_exg_thresh,
        cfg.young_min_blob_m2, cfg.young_compact_size_m2, cfg.young_circularity_thresh,
        row_proximity_mask=row_mask, exclude_mask=tall_mask,
    )
    young_meta.update(ym)

    validation = None
    if tree_locations_path is not None:
        points_gdf = gpd.read_file(tree_locations_path, layer=tree_locations_layer) if tree_locations_layer \
            else gpd.read_file(tree_locations_path)
        if points_gdf.crs != stack["crs"]:
            points_gdf = points_gdf.to_crs(stack["crs"])
        validation = validate_against_tree_locations(
            young_mask, mature_mask, stack["transform"], points_gdf, cfg.tree_location_match_radius_m,
        )

    px_area = pixel_area_m2(stack["transform"])
    plot_area_m2 = valid.sum() * px_area
    mature_area_m2 = mature_mask[valid].sum() * px_area
    other_veg_area_m2 = other_veg_mask[valid].sum() * px_area
    young_area_m2 = young_mask[valid].sum() * px_area
    total_area_m2 = mature_area_m2 + young_area_m2 + other_veg_area_m2

    n_young_blobs = int(label(young_mask, connectivity=2).max())

    result = {
        "plot_name": plot_name,
        "plot_area_m2": float(plot_area_m2),
        "plot_area_ha": float(plot_area_m2 / 10000),
        "kpi1_total_canopy_pct": float(100 * total_area_m2 / plot_area_m2),
        "kpi1_total_canopy_m2": float(total_area_m2),
        "kpi2_mature_canopy_pct": float(100 * mature_area_m2 / plot_area_m2),
        "kpi2_mature_canopy_m2": float(mature_area_m2),
        "n_mature_tree_blobs": int(label(mature_mask, connectivity=2).max()),
        "kpi3_young_canopy_pct": float(100 * young_area_m2 / plot_area_m2),
        "kpi3_young_canopy_m2": float(young_area_m2),
        "n_young_tree_blobs": n_young_blobs,
        "other_vegetation_pct": float(100 * other_veg_area_m2 / plot_area_m2),
        "other_vegetation_m2": float(other_veg_area_m2),
        "n_other_vegetation_blobs": shape_meta["n_other_vegetation_blobs"],
        "young_tree_detection": young_meta,
        "tree_location_validation": validation,
        "validation_status": "UNVALIDATED - no field-verified ground-truth points supplied. "
                              "The tree-location file (if used) gives a recall proxy only, not "
                              "precision. Do not present these KPIs as final numbers on the "
                              "dashboard until validated against reference points (see CLAUDE.md).",
        "config": asdict(cfg),
    }

    if outdir is not None:
        _export_outputs(
            Path(outdir), stack, mature_mask, other_veg_mask, young_mask, result,
        )

    return result


def _export_outputs(outdir: Path, stack, mature_mask, other_veg_mask, young_mask, result):
    outdir.mkdir(parents=True, exist_ok=True)
    transform, crs = stack["transform"], stack["crs"]
    h, w = mature_mask.shape

    total_mask = mature_mask | young_mask | other_veg_mask  # matches KPI1

    mask_stack = np.stack([
        total_mask.astype(np.uint8),
        mature_mask.astype(np.uint8),
        young_mask.astype(np.uint8),
        other_veg_mask.astype(np.uint8),
    ])
    with rasterio.open(
        outdir / "canopy_mask.tif", "w", driver="GTiff", height=h, width=w, count=4,
        dtype=np.uint8, crs=crs, transform=transform, compress="deflate", nodata=0,
    ) as dst:
        dst.write(mask_stack)
        dst.descriptions = ("kpi1_total", "kpi2_mature", "kpi3_young_candidate", "other_vegetation")

    gpkg_path = outdir / "canopy_layers.gpkg"
    if gpkg_path.exists():
        gpkg_path.unlink()
    any_layer = False
    for layer_name, m in [
        ("mature_trees", mature_mask),
        ("young_tree_candidates", young_mask),
        ("other_vegetation", other_veg_mask),
    ]:
        shapes = [
            (geom, 1) for geom, val in rasterio.features.shapes(m.astype(np.uint8), mask=m, transform=transform)
            if val == 1
        ]
        if not shapes:
            continue
        import shapely.geometry
        gdf = gpd.GeoDataFrame(
            geometry=[shapely.geometry.shape(g) for g, _ in shapes], crs=crs,
        )
        gdf["area_m2"] = gdf.geometry.area
        gdf["blob_id"] = range(1, len(gdf) + 1)
        gdf.to_file(gpkg_path, layer=layer_name, driver="GPKG")
        any_layer = True

    _export_detected_trees(outdir, transform, crs, mature_mask, young_mask, other_veg_mask)

    with open(outdir / "summary.json", "w") as f:
        json.dump(result, f, indent=2)

    _save_overlay_png(outdir / "overlay.png", stack, mature_mask, other_veg_mask, young_mask)
    export_web_layers(outdir, stack, mature_mask, other_veg_mask, young_mask)


def _export_detected_trees(outdir: Path, transform, crs, mature_mask, young_mask, other_veg_mask):
    """One point per detected blob (its centroid), classed as Mature tree /
    Young tree / Other. Much lighter than the polygon layers and easier to
    count/import elsewhere -- particularly useful on plots dominated by
    thousands of tiny young-tree candidate blobs (e.g. plot_3)."""
    import shapely.geometry

    px_area = pixel_area_m2(transform)
    records = []
    for class_label, mask in [
        ("Mature tree", mature_mask),
        ("Young tree", young_mask),
        ("Other", other_veg_mask),
    ]:
        lbl = label(mask, connectivity=2)
        for prop in regionprops(lbl):
            y, x = prop.centroid
            mx, my = transform * (x, y)
            records.append({
                "class": class_label,
                "area_m2": prop.area * px_area,
                "geometry": shapely.geometry.Point(mx, my),
            })

    if not records:
        return
    gdf = gpd.GeoDataFrame(records, crs=crs)
    gdf.insert(0, "tree_id", range(1, len(gdf) + 1))
    gdf.to_file(outdir / "detected_trees.gpkg", layer="detected_trees", driver="GPKG")


def _save_overlay_png(path, stack, mature_mask, other_veg_mask, young_mask):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = np.moveaxis(stack["rgb"], 0, -1)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(rgb)
    overlay = np.zeros((*mature_mask.shape, 4))
    overlay[other_veg_mask] = [1, 0.6, 0, 0.6]  # orange = other vegetation (mato/hedge)
    overlay[young_mask] = [0, 1, 1, 0.6]        # cyan = potential young trees
    overlay[mature_mask] = [1, 0, 0, 0.6]       # red = mature/identifiable trees
    ax.imshow(overlay)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
