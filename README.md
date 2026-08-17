# Canopy Cover Detection

Pure-Python pipeline that computes canopy cover KPIs for an agroforestry
plot from drone survey products (orthophoto, DSM, DTM, plot boundary). No
QGIS dependency — built on `rasterio`, `geopandas`, and `scikit-image`.

Built for plots where most of the planted trees are too young/small to be
reliably identified in RGB imagery or a canopy height model, so the
pipeline treats "clearly a tree" and "possibly a young tree" as two
separate, explicitly-labeled outputs rather than forcing a single
detection into one number.

## KPIs

- **KPI1 — Total Canopy** = KPI2 + KPI3.
- **KPI2 — Mature Canopy**: trees clearly identifiable as trees, any
  species (height + greenness + compact shape).
- **KPI3 — Young Canopy**: spectral *candidate* blobs too small/short to
  be unambiguous — reported as both a cover % and a blob count.
- **Other vegetation** (scrub/hedge/"mato"): tracked and exported as its
  own layer, but **not** a KPI and excluded from KPI1.

All KPIs are marked `UNVALIDATED` in every output until checked against
field-verified reference points — see `CLAUDE.md` in a full checkout for
the detailed methodology and validation notes (not included in this public
repo).

## Install

```bash
python -m venv venv
venv/Scripts/pip install -r requirements.txt   # Windows
# source venv/bin/activate && pip install -r requirements.txt   # macOS/Linux
```

## Web UI (recommended)

```bash
venv/Scripts/python app.py   # Windows
# venv/bin/python app.py     # macOS/Linux
```

Then open **http://localhost:8050**. Drop all your files into the one
basket at once — DSM, DTM, orthophoto, land boundary (required), and
optionally a planting-row layer (improves young-tree detection accuracy).
Files are sorted automatically by filename/content, each with an editable
role dropdown in case anything gets misclassified. The **Run Canopy Cover
Detection** button stays disabled with a warning until every required
input is covered; once you click it, the overlay + 3 KPIs appear alongside
download links for every output. Everything runs locally on your machine —
files are never uploaded anywhere else.

Accepted vector formats: `.gpkg`, `.geojson`. Shapefiles are supported too —
drop the whole bundle (`.shp` + `.shx` + `.dbf` + `.prj`) into the basket
together.

## Library / CLI usage

```python
from canopy_cover import compute_canopy_cover, CanopyConfig

result = compute_canopy_cover(
    orthophoto_path="path/to/orthophoto.tif",
    dsm_path="path/to/dsm.tif",
    dtm_path="path/to/dtm.tif",
    boundary_path="path/to/boundary.gpkg",
    planting_rows_path="path/to/rows.gpkg",   # optional, improves young-tree detection
    tree_locations_path="path/to/trees.gpkg", # optional, recall check only
    config=CanopyConfig(),                    # thresholds are tunable, see pipeline.py
    outdir="outputs/my_plot",
    plot_name="my_plot",
)
```

Or via the CLI wrapper (edit `PLOT_CONFIG` in `canopy_cover/cli.py` to
point at your own files):

```bash
python -m canopy_cover.cli --plot plot_1
```

Each run writes to `outputs/<plot_name>/`:

| File | Contents |
|---|---|
| `canopy_mask.tif` | 4-band GeoTIFF: total / mature / young-candidate / other-vegetation |
| `canopy_layers.gpkg` | Vector layers: `mature_trees`, `young_tree_candidates`, `other_vegetation` |
| `canopy_polygons.geojson` | Same polygons as GeoJSON |
| `summary.json` | KPI values, blob counts, detection metadata, validation status |
| `overlay.png` | Orthophoto with detection layers overlaid |

## Project layout

```
canopy_cover/
  raster_utils.py   # aligns orthophoto/DSM/DTM onto a common grid, computes CHM
  vegetation_index.py  # ExG, VARI (RGB-only indices)
  masks.py          # mature-tree and young-tree-candidate mask construction
  validate.py        # recall check against a tracked tree-location file
  pipeline.py        # compute_canopy_cover(...) orchestration + exports
  cli.py              # CLI wrapper
scripts/eda.py        # exploratory data analysis (CHM/ExG histograms) — run before tuning thresholds on a new plot
app.py                 # drag-and-drop web UI (Dash), see "Web UI" above
```

Each run of the web UI writes to a per-session temp folder (`.runs/`,
cleared on every restart of `app.py`) rather than `outputs/`, which is only
used by the CLI/library path.

## Status / roadmap

Pipeline validated on three real plots via a recall check against tracked
tree locations (not yet a full precision/recall validation against
confirmed field ground truth) — see `CLAUDE.md` in a full checkout.
