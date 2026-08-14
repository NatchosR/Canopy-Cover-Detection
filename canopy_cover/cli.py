"""CLI wrapper: python -m canopy_cover.cli --plot plot_1"""
import argparse
import json
from pathlib import Path

from .pipeline import compute_canopy_cover, CanopyConfig

PLOT_CONFIG = {
    "plot_1": dict(
        orthophoto="data/plot_1/plot1_orthophoto_GCP_32629.tif",
        dsm="data/plot_1/plot1_dsm_GCP_32629.tif",
        dtm="data/plot_1/plot1_dtm_GCP_32629.tif",
        boundary="data/plot_1/plot1_boundary.gpkg",
        boundary_layer="land",
        planting_rows="data/plot_1/line_layer.gpkg",
        planting_rows_layer="new_line",
        tree_locations="data/plot_1/plot1_tree_locations.gpkg",
        tree_locations_layer="plot1_tree_locations",
    ),
    "plot_2": dict(
        orthophoto="data/plot_2/plot2_orthophoto_32629.tif",
        dsm="data/plot_2/plot2_dsm_32629.tif",
        dtm="data/plot_2/plot2_dtm_32629.tif",
        boundary="data/plot_2/plot2_boundary.gpkg",
        boundary_layer=None,
        planting_rows=None,
        planting_rows_layer=None,
        tree_locations=None,
        tree_locations_layer=None,
    ),
    "plot_3": dict(
        orthophoto="data/plot_3/plot3_orthophoto_32629.tif",
        dsm="data/plot_3/plot3_dsm_32629.tif",
        dtm="data/plot_3/plot3_dtm_32629.tif",
        boundary="data/plot_3/plot3_boundary.gpkg",
        boundary_layer=None,
        planting_rows=None,
        planting_rows_layer=None,
        tree_locations=None,
        tree_locations_layer=None,
    ),
}


def main():
    parser = argparse.ArgumentParser(description="Compute canopy cover KPIs for a plot.")
    parser.add_argument("--plot", required=True, choices=PLOT_CONFIG.keys())
    parser.add_argument("--no-rows", action="store_true", help="Ignore planting rows even if available.")
    parser.add_argument("--no-tree-locations", action="store_true", help="Skip tree-location validation even if available.")
    parser.add_argument("--outdir", default=None, help="Defaults to outputs/<plot>")
    args = parser.parse_args()

    cfg = PLOT_CONFIG[args.plot]
    planting_rows = None if args.no_rows else cfg["planting_rows"]
    tree_locations = None if args.no_tree_locations else cfg["tree_locations"]

    variant = "_no_rows" if args.no_rows and cfg["planting_rows"] else ""
    outdir = args.outdir or f"outputs/{args.plot}{variant}"

    result = compute_canopy_cover(
        cfg["orthophoto"], cfg["dsm"], cfg["dtm"], cfg["boundary"],
        boundary_layer=cfg["boundary_layer"],
        planting_rows_path=planting_rows, planting_rows_layer=cfg["planting_rows_layer"],
        tree_locations_path=tree_locations, tree_locations_layer=cfg["tree_locations_layer"],
        config=CanopyConfig(),
        outdir=outdir,
        plot_name=args.plot,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
