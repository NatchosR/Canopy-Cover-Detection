"""Exploratory data analysis: CHM noise floor, height distribution, ExG distribution.

Run before picking any detection thresholds. Produces diagnostic PNGs + printed stats.
Usage: python scripts/eda.py plot_1
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio.features
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from canopy_cover.raster_utils import load_aligned_stack, compute_chm

PLOT_CONFIG = {
    "plot_1": dict(
        ortho="data/plot_1/plot1_orthophoto_GCP_32629.tif",
        dsm="data/plot_1/plot1_dsm_GCP_32629.tif",
        dtm="data/plot_1/plot1_dtm_GCP_32629.tif",
        boundary="data/plot_1/plot1_boundary.gpkg",
        boundary_layer="land",
    ),
    "plot_2": dict(
        ortho="data/plot_2/plot2_orthophoto_32629.tif",
        dsm="data/plot_2/plot2_dsm_32629.tif",
        dtm="data/plot_2/plot2_dtm_32629.tif",
        boundary="data/plot_2/plot2_boundary.gpkg",
        boundary_layer=None,
    ),
}


def rasterize_boundary(boundary_path, layer, transform, shape):
    gdf = gpd.read_file(boundary_path, layer=layer) if layer else gpd.read_file(boundary_path)
    mask = rasterio.features.rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=shape, transform=transform, fill=0, dtype=np.uint8,
    )
    return mask.astype(bool)


def main(plot_key):
    cfg = PLOT_CONFIG[plot_key]
    outdir = Path("outputs") / plot_key
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"=== {plot_key} ===")
    stack = load_aligned_stack(cfg["ortho"], cfg["dsm"], cfg["dtm"])
    chm, valid = compute_chm(stack["dsm"], stack["dtm"], stack["dsm_mask"], stack["dtm_mask"])

    boundary_mask = rasterize_boundary(cfg["boundary"], cfg["boundary_layer"], stack["transform"], chm.shape)
    in_plot = valid & boundary_mask
    print(f"raster shape: {chm.shape}, valid px: {valid.sum()}, in-boundary valid px: {in_plot.sum()}")

    chm_vals = chm[in_plot]
    print("\n--- CHM stats (within boundary) ---")
    for p in [0, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.5, 99.9, 100]:
        print(f"  p{p:>5}: {np.percentile(chm_vals, p):.3f} m")
    print(f"  mean: {chm_vals.mean():.3f}  std: {chm_vals.std():.3f}")
    print(f"  frac < 0:    {(chm_vals < 0).mean()*100:.2f}%")
    print(f"  frac < 0.1:  {(chm_vals < 0.1).mean()*100:.2f}%")
    print(f"  frac < 0.2:  {(chm_vals < 0.2).mean()*100:.2f}%")
    print(f"  frac < 0.3:  {(chm_vals < 0.3).mean()*100:.2f}%")
    print(f"  frac < 0.5:  {(chm_vals < 0.5).mean()*100:.2f}%")
    print(f"  frac > 1:    {(chm_vals > 1).mean()*100:.2f}%")
    print(f"  frac > 2:    {(chm_vals > 2).mean()*100:.2f}%")
    print(f"  frac > 3:    {(chm_vals > 3).mean()*100:.2f}%")

    # ExG = 2G - R - B, on normalized [0,1] RGB
    rgb = stack["rgb"].astype(np.float32) / 255.0
    r, g, b = rgb[0], rgb[1], rgb[2]
    exg = 2 * g - r - b
    exg_vals = exg[in_plot]
    print("\n--- ExG stats (within boundary) ---")
    for p in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
        print(f"  p{p:>5}: {np.percentile(exg_vals, p):.4f}")

    # Diagnostic plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    ax = axes[0, 0]
    ax.imshow(np.moveaxis(stack["rgb"], 0, -1))
    ax.set_title(f"{plot_key} orthophoto")
    ax.axis("off")

    ax = axes[0, 1]
    chm_disp = np.where(valid, chm, np.nan)
    im = ax.imshow(np.clip(chm_disp, -0.5, 5), cmap="viridis")
    ax.set_title("CHM (clipped -0.5 to 5 m)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 2]
    im = ax.imshow(np.clip(chm_disp, -0.5, 1.5), cmap="RdYlGn")
    ax.set_title("CHM (clipped -0.5 to 1.5 m, low-height detail)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    ax.hist(chm_vals, bins=200, range=(-1, 5))
    ax.set_yscale("log")
    ax.set_title("CHM histogram (log y)")
    ax.set_xlabel("height (m)")
    ax.axvline(0, color="r", lw=0.5)

    ax = axes[1, 1]
    ax.hist(chm_vals, bins=200, range=(-0.5, 1.5))
    ax.set_yscale("log")
    ax.set_title("CHM histogram, low-height zoom (log y)")
    ax.set_xlabel("height (m)")
    ax.axvline(0, color="r", lw=0.5)

    ax = axes[1, 2]
    exg_disp = np.where(valid, exg, np.nan)
    im = ax.imshow(exg_disp, cmap="RdYlGn", vmin=-0.3, vmax=0.3)
    ax.set_title("ExG")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    outpath = outdir / "eda_overview.png"
    plt.savefig(outpath, dpi=130)
    print(f"\nSaved {outpath}")

    # Save raw arrays for reuse in later steps (avoid recompute)
    np.savez_compressed(
        outdir / "eda_arrays.npz",
        chm=chm, valid=valid, boundary_mask=boundary_mask, exg=exg,
    )
    print(f"Saved {outdir / 'eda_arrays.npz'}")


if __name__ == "__main__":
    plot = sys.argv[1] if len(sys.argv) > 1 else "plot_1"
    main(plot)
