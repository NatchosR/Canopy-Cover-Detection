"""Canopy cover detection -- standalone Web UI.

Run locally: venv/Scripts/python app.py, then open http://localhost:8050

The user drags and drops their own DSM, DTM, orthophoto, and land-boundary
files (planting rows optional), clicks "Run Canopy Cover Detection", and
gets the overlay + 3 KPIs plus every output available for download. No
precomputed/example data is involved -- this is the distributable UI
described in CLAUDE.md's "Next session (Monday)" plan.
"""
import base64
import re
import shutil
import uuid
from pathlib import Path

import dash
import flask
from dash import ALL, Dash, Input, Output, State, dcc, html
from werkzeug.utils import secure_filename

from canopy_cover.pipeline import CanopyConfig, compute_canopy_cover

RUNS_DIR = Path(".runs")
# Fresh start each launch -- these are ephemeral per-session working files, not
# meant to persist across restarts of the app.
if RUNS_DIR.exists():
    shutil.rmtree(RUNS_DIR)
RUNS_DIR.mkdir(exist_ok=True)

REQUIRED_INPUTS = ["dsm", "dtm", "ortho", "boundary"]
OPTIONAL_INPUTS = ["rows"]
UPLOAD_KEYS = REQUIRED_INPUTS + OPTIONAL_INPUTS
INPUT_LABELS = {
    "dsm": "DSM — Digital Surface Model",
    "dtm": "DTM — Digital Terrain Model",
    "ortho": "Orthophoto (RGB)",
    "boundary": "Land boundary",
    "rows": "Planting rows (optional)",
}
# Boundary/rows accept multi-file drops (a .shp needs its .shx/.dbf/.prj siblings).
MULTI_FILE_KEYS = {"boundary", "rows"}
VECTOR_EXT_PRIORITY = (".gpkg", ".geojson", ".json", ".shp")

DOWNLOADABLE_FILES = [
    ("canopy_mask.tif", "Canopy mask (GeoTIFF, 4 bands)"),
    ("canopy_layers.gpkg", "Canopy layers (GeoPackage)"),
    ("canopy_polygons.geojson", "Canopy polygons (GeoJSON)"),
    ("summary.json", "KPI summary (JSON)"),
    ("overlay.png", "Overlay preview (PNG)"),
]
WEB_ASSET_FILES = {"web_base_rgb.png", "web_mature_overlay.png", "web_young_overlay.png", "web_other_overlay.png"}
ALLOWED_RESULT_FILES = {f for f, _ in DOWNLOADABLE_FILES} | WEB_ASSET_FILES

SESSION_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _session_dir(session_id: str) -> Path:
    if not SESSION_ID_RE.match(session_id or ""):
        raise ValueError("invalid session id")
    d = RUNS_DIR / session_id
    (d / "inputs").mkdir(parents=True, exist_ok=True)
    (d / "results").mkdir(parents=True, exist_ok=True)
    return d


def _save_upload_slot(dest_dir: Path, contents, filenames) -> list[str]:
    """Decode and write one or more uploaded files to dest_dir. Skips the
    write if the same set of filenames is already on disk (Dash re-delivers
    unchanged `contents` for every matched component whenever any sibling
    upload fires), and clears stale files when the slot is replaced."""
    if not isinstance(contents, list):
        contents, filenames = [contents], [filenames]
    safe_names = [secure_filename(fn) or "file" for fn in filenames]

    existing = {p.name for p in dest_dir.glob("*")} if dest_dir.exists() else set()
    if existing != set(safe_names):
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        for content, name in zip(contents, safe_names):
            _, b64data = content.split(",", 1)
            (dest_dir / name).write_bytes(base64.b64decode(b64data))

    return [str(dest_dir / name) for name in safe_names]


def _pick_vector_path(paths) -> str:
    """Given one or more uploaded vector files (possibly a shapefile bundle),
    pick the path to actually hand to geopandas."""
    if isinstance(paths, str):
        return paths
    paths = [Path(p) for p in paths]
    for ext in VECTOR_EXT_PRIORITY:
        for p in paths:
            if p.suffix.lower() == ext:
                return str(p)
    return str(paths[0])


app = Dash(__name__, title="Canopy Cover Detection", suppress_callback_exceptions=True)
server = app.server
server.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # drone rasters can be large


@server.route("/results/<session_id>/<path:filename>")
def serve_result(session_id, filename):
    if not SESSION_ID_RE.match(session_id) or filename not in ALLOWED_RESULT_FILES:
        flask.abort(404)
    directory = (RUNS_DIR / session_id / "results").resolve()
    if not (directory / filename).exists():
        flask.abort(404)
    return flask.send_from_directory(directory, filename)


# ---------------------------------------------------------------- components

def upload_slot(key):
    required = key in REQUIRED_INPUTS
    return html.Div(
        [
            html.Div(
                [
                    INPUT_LABELS[key],
                    html.Span(" required" if required else " optional", className="req-tag" if required else "opt-tag"),
                ],
                className="upload-label",
            ),
            dcc.Upload(
                id={"type": "upload", "key": key},
                children=html.Div("Drag & drop here, or click to browse", id={"type": "upload-status", "key": key}),
                className="upload-zone",
                multiple=key in MULTI_FILE_KEYS,
            ),
        ],
        className="upload-slot",
    )


def kpi_card(title, value, subtitle, color):
    return html.Div(
        [
            html.Div(title, className="kpi-title"),
            html.Div(value, className="kpi-value", style={"color": color}),
            html.Div(subtitle, className="kpi-subtitle"),
        ],
        className="kpi-card",
    )


def build_kpi_row(summary):
    return [
        kpi_card("KPI1 — Total Canopy", f"{summary['kpi1_total_canopy_pct']:.2f}%",
                  f"{summary['kpi1_total_canopy_m2']:.0f} m² · mature + young", "#1b4332"),
        kpi_card("KPI2 — Mature Trees", f"{summary['kpi2_mature_canopy_pct']:.2f}%",
                  f"{summary['n_mature_tree_blobs']} blobs", "#c1121f"),
        kpi_card("KPI3 — Young Tree Candidates", f"{summary['kpi3_young_canopy_pct']:.2f}%",
                  f"{summary['n_young_tree_blobs']} blobs (potential trees)", "#0891a8"),
        kpi_card("Other Vegetation (mato)", f"{summary['other_vegetation_pct']:.2f}%",
                  f"{summary['n_other_vegetation_blobs']} blobs — not a KPI", "#e07a00"),
    ]


def build_validation_banner(summary):
    return html.Div(
        [html.Strong("Validation status: "), summary.get("validation_status", "unvalidated")],
        className="validation-banner warn",
    )


def build_viewer(session_id, layer_values):
    def vis(name):
        return "block" if name in layer_values else "none"

    base = f"/results/{session_id}"
    return [
        html.Img(src=f"{base}/web_base_rgb.png", className="layer-img"),
        html.Img(src=f"{base}/web_mature_overlay.png", className="overlay-img", style={"display": vis("mature")}),
        html.Img(src=f"{base}/web_young_overlay.png", className="overlay-img", style={"display": vis("young")}),
        html.Img(src=f"{base}/web_other_overlay.png", className="overlay-img", style={"display": vis("other")}),
    ]


def build_downloads(session_id):
    return [
        html.A(label, href=f"/results/{session_id}/{fname}", download=fname, className="download-link")
        for fname, label in DOWNLOADABLE_FILES
    ]


def build_results_view(session_id, summary):
    default_layers = ["mature", "young", "other"]
    return html.Div(
        [
            build_validation_banner(summary),
            html.Div(build_kpi_row(summary), className="kpi-row"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Layers", className="panel-title"),
                            dcc.Checklist(
                                id="layer-toggle",
                                options=[
                                    {"label": " Mature trees (KPI2)", "value": "mature"},
                                    {"label": " Young tree candidates (KPI3)", "value": "young"},
                                    {"label": " Other vegetation (mato)", "value": "other"},
                                ],
                                value=default_layers,
                                className="layer-checklist",
                            ),
                            html.Div(
                                [
                                    html.Div([html.Span(className="legend-swatch", style={"background": "#dc2828"}), "Mature"]),
                                    html.Div([html.Span(className="legend-swatch", style={"background": "#00dcdc"}), "Young candidate"]),
                                    html.Div([html.Span(className="legend-swatch", style={"background": "#f09600"}), "Other veg."]),
                                ],
                                className="legend",
                            ),
                            html.Div("Downloads", className="panel-title", style={"marginTop": "22px"}),
                            html.Div(build_downloads(session_id)),
                        ],
                        className="sidebar",
                    ),
                    html.Div(
                        html.Div(id="image-viewer", className="image-viewer", children=build_viewer(session_id, default_layers)),
                        className="viewer-container",
                    ),
                ],
                className="main-row",
            ),
            dcc.Store(id="results-session-id", data=session_id),
        ]
    )


# --------------------------------------------------------------------- layout

def build_layout():
    session_id = str(uuid.uuid4())
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Canopy Cover Detection"),
                    html.P(
                        "Drop your drone survey products below, run detection, and download the results. "
                        "Everything runs locally -- your files never leave this machine.",
                        className="subtitle",
                    ),
                ],
                className="header",
            ),
            html.Div([upload_slot(k) for k in UPLOAD_KEYS], className="upload-grid"),
            html.Div(
                [
                    html.Button("Run Canopy Cover Detection", id="run-button", n_clicks=0, className="run-button"),
                    html.Div(id="run-error"),
                ],
                className="run-row",
            ),
            dcc.Loading(html.Div(id="results-container"), type="circle", color="#2c7a4b"),
            dcc.Store(id="session-id", data=session_id),
            dcc.Store(id="upload-paths", data={}),
        ],
        className="app-container",
    )


app.layout = build_layout


# ------------------------------------------------------------------ callbacks

@app.callback(
    Output({"type": "upload-status", "key": ALL}, "children"),
    Output("upload-paths", "data"),
    Input({"type": "upload", "key": ALL}, "contents"),
    State({"type": "upload", "key": ALL}, "filename"),
    State({"type": "upload", "key": ALL}, "id"),
    State("session-id", "data"),
    State("upload-paths", "data"),
    prevent_initial_call=True,
)
def on_upload(all_contents, all_filenames, all_ids, session_id, paths_store):
    paths_store = dict(paths_store or {})
    session = _session_dir(session_id)
    statuses = []
    for contents, filenames, id_ in zip(all_contents, all_filenames, all_ids):
        key = id_["key"]
        if contents is None:
            if key in paths_store:
                saved = paths_store[key]
                names = [Path(p).name for p in saved] if isinstance(saved, list) else [Path(saved).name]
                statuses.append(f"✓ {', '.join(names)}")
            else:
                statuses.append("Drag & drop here, or click to browse")
            continue
        saved_paths = _save_upload_slot(session / "inputs" / key, contents, filenames)
        paths_store[key] = saved_paths if key in MULTI_FILE_KEYS else saved_paths[0]
        statuses.append(f"✓ {', '.join(Path(p).name for p in saved_paths)}")
    return statuses, paths_store


@app.callback(
    Output("results-container", "children"),
    Output("run-error", "children"),
    Input("run-button", "n_clicks"),
    State("upload-paths", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_run(n_clicks, paths_store, session_id):
    paths_store = paths_store or {}
    missing = [INPUT_LABELS[k] for k in REQUIRED_INPUTS if k not in paths_store]
    if missing:
        return dash.no_update, html.Div(f"Missing required input(s): {', '.join(missing)}.", className="run-error")

    session = _session_dir(session_id)
    results_dir = session / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)

    try:
        result = compute_canopy_cover(
            orthophoto_path=paths_store["ortho"],
            dsm_path=paths_store["dsm"],
            dtm_path=paths_store["dtm"],
            boundary_path=_pick_vector_path(paths_store["boundary"]),
            planting_rows_path=_pick_vector_path(paths_store["rows"]) if "rows" in paths_store else None,
            config=CanopyConfig(),
            outdir=str(results_dir),
            plot_name=session_id,
        )
    except Exception as exc:  # surface to the user instead of a 500 page
        return dash.no_update, html.Div(f"Detection failed: {exc}", className="run-error")

    return build_results_view(session_id, result), None


@app.callback(
    Output("image-viewer", "children"),
    Input("layer-toggle", "value"),
    State("results-session-id", "data"),
    prevent_initial_call=True,
)
def on_layer_toggle(layer_values, session_id):
    if not session_id:
        return dash.no_update
    return build_viewer(session_id, layer_values or [])


if __name__ == "__main__":
    app.run(debug=True, port=8050)
