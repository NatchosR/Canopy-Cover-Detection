"""Canopy cover detection -- standalone Web UI.

Run locally: venv/Scripts/python app.py, then open http://localhost:8050

The user drags and drops ALL their files (DSM, DTM, orthophoto, boundary,
optional planting rows) into one basket at once. Files are auto-classified
by filename/content and shown with an editable role dropdown so
misclassifications can be corrected. The Run button is disabled until every
required role is covered. Results reuse the KPI cards / layered viewer /
download links built earlier, now driven by the basket instead of five
separate upload slots.
"""
import base64
import re
import shutil
import uuid
from pathlib import Path

import dash
import flask
import rasterio
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate
from werkzeug.utils import secure_filename

from canopy_cover.pipeline import CanopyConfig, compute_canopy_cover

RUNS_DIR = Path(".runs")
# Fresh start each launch -- these are ephemeral per-session working files, not
# meant to persist across restarts of the app.
if RUNS_DIR.exists():
    shutil.rmtree(RUNS_DIR)
RUNS_DIR.mkdir(exist_ok=True)

ROLE_LABELS = {
    "dsm": "DSM",
    "dtm": "DTM",
    "ortho": "Orthophoto",
    "boundary": "Boundary",
    "rows": "Planting rows",
}
REQUIRED_ROLES = ["dsm", "dtm", "ortho", "boundary"]
OPTIONAL_ROLES = ["rows"]
ROLE_DROPDOWN_OPTIONS = [{"label": "— ignore —", "value": ""}] + [
    {"label": label, "value": role} for role, label in ROLE_LABELS.items()
]

VECTOR_SIDECAR_EXT = {".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx"}
VECTOR_STANDALONE_EXT = {".gpkg", ".geojson", ".json"}
RASTER_EXT = {".tif", ".tiff"}
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
    (d / "inputs" / "_raw").mkdir(parents=True, exist_ok=True)
    (d / "results").mkdir(parents=True, exist_ok=True)
    return d


def _pick_vector_path(paths) -> str:
    """Given one or more files (possibly a shapefile bundle), pick the path
    to actually hand to geopandas."""
    paths = [Path(p) for p in paths]
    for ext in VECTOR_EXT_PRIORITY:
        for p in paths:
            if p.suffix.lower() == ext:
                return str(p)
    return str(paths[0])


def _guess_raster_role(filename: str, path: Path):
    name = filename.lower()
    if "dsm" in name:
        return "dsm"
    if "dtm" in name:
        return "dtm"
    if any(k in name for k in ("ortho", "rgb", "photo")):
        return "ortho"
    # Fallback: DSM/DTM are single-band float rasters, orthophotos are
    # multi-band uint8 -- this alone can't distinguish DSM from DTM.
    try:
        with rasterio.open(path) as src:
            if src.count >= 3 and src.dtypes[0] == "uint8":
                return "ortho"
    except Exception:
        pass
    return None


def _guess_vector_role(filename: str):
    name = filename.lower()
    if any(k in name for k in ("row", "line", "planting")):
        return "rows"
    if any(k in name for k in ("boundary", "bound", "land", "parcel", "plot")):
        return "boundary"
    return None


def _scan_and_classify(raw_dir: Path, overrides: dict) -> list:
    """Rebuild the basket item list from whatever files currently sit in
    raw_dir, grouping shapefile sidecars and applying any user overrides."""
    files = sorted(raw_dir.glob("*")) if raw_dir.exists() else []
    shp_groups = {}
    standalone = []
    for p in files:
        if p.suffix.lower() in VECTOR_SIDECAR_EXT:
            shp_groups.setdefault(p.stem.lower(), []).append(p)
        else:
            standalone.append(p)

    items = []
    for paths in shp_groups.values():
        shp = next((p for p in paths if p.suffix.lower() == ".shp"), paths[0])
        key = shp.name
        role = overrides[key] if key in overrides else _guess_vector_role(shp.name)
        items.append({"key": key, "filenames": [p.name for p in paths], "paths": [str(p) for p in paths], "role": role})

    for p in standalone:
        ext = p.suffix.lower()
        key = p.name
        if ext in RASTER_EXT:
            role = overrides[key] if key in overrides else _guess_raster_role(p.name, p)
        elif ext in VECTOR_STANDALONE_EXT:
            role = overrides[key] if key in overrides else _guess_vector_role(p.name)
        else:
            role = overrides.get(key)
        items.append({"key": key, "filenames": [p.name], "paths": [str(p)], "role": role})

    items.sort(key=lambda it: it["filenames"][0].lower())
    return items


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

def item_card(item):
    return html.Div(
        [
            html.Div(", ".join(item["filenames"]), className="item-filename"),
            dcc.Dropdown(
                id={"type": "role-select", "key": item["key"]},
                options=ROLE_DROPDOWN_OPTIONS,
                value=item["role"] or "",
                clearable=False,
                className="role-select",
            ),
            html.Button("×", id={"type": "remove-item", "key": item["key"]}, n_clicks=0, className="remove-item-btn"),
        ],
        className="item-card" if item["role"] else "item-card unassigned",
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
                        "Drop everything below, run detection, and download the results. "
                        "Everything runs locally -- your files never leave this machine.",
                        className="subtitle",
                    ),
                ],
                className="header",
            ),
            html.Div(
                [
                    dcc.Upload(
                        id="basket-upload",
                        children=html.Div(
                            [
                                html.Div("Drop all your files here", className="basket-title"),
                                html.Div(
                                    "DSM, DTM, orthophoto, and boundary are required; a planting-row layer is "
                                    "optional. Drop them all at once, or add more later -- files are sorted "
                                    "automatically and you can fix any mistakes below.",
                                    className="basket-subtitle",
                                ),
                            ]
                        ),
                        className="basket-zone",
                        multiple=True,
                    ),
                    html.Div(
                        [
                            html.Div("Detected files", className="panel-title-inline"),
                            html.Button("Clear all", id="clear-basket-btn", n_clicks=0, className="clear-btn"),
                        ],
                        className="basket-list-header",
                    ),
                    html.Div(id="basket-list", className="basket-list", children=html.Div("No files added yet.", className="basket-empty")),
                    html.Div(id="missing-warning"),
                ],
                className="basket-container",
            ),
            html.Div(
                [
                    html.Button("Run Canopy Cover Detection", id="run-button", n_clicks=0, className="run-button", disabled=True),
                    html.Div(id="run-error"),
                ],
                className="run-row",
            ),
            dcc.Loading(html.Div(id="results-container"), type="circle", color="#2c7a4b"),
            dcc.Store(id="session-id", data=session_id),
            dcc.Store(id="raw-version", data=0),
            dcc.Store(id="role-overrides", data={}),
            dcc.Store(id="basket-items", data=[]),
        ],
        className="app-container",
    )


app.layout = build_layout


# ------------------------------------------------------------------ callbacks

@app.callback(
    Output("raw-version", "data"),
    Input("basket-upload", "contents"),
    State("basket-upload", "filename"),
    State("session-id", "data"),
    State("raw-version", "data"),
    prevent_initial_call=True,
)
def on_basket_drop(contents_list, filenames_list, session_id, version):
    if not contents_list:
        raise PreventUpdate
    if not isinstance(contents_list, list):
        contents_list, filenames_list = [contents_list], [filenames_list]

    raw_dir = _session_dir(session_id) / "inputs" / "_raw"
    for content, filename in zip(contents_list, filenames_list):
        safe_name = secure_filename(filename) or "file"
        path = raw_dir / safe_name
        if path.exists():
            continue  # already have this exact file; Dash re-delivers old contents on rerender
        _, b64data = content.split(",", 1)
        path.write_bytes(base64.b64decode(b64data))

    return (version or 0) + 1


@app.callback(
    Output("raw-version", "data", allow_duplicate=True),
    Input({"type": "remove-item", "key": ALL}, "n_clicks"),
    State("session-id", "data"),
    State("raw-version", "data"),
    prevent_initial_call=True,
)
def on_remove_item(n_clicks_list, session_id, version):
    triggered = ctx.triggered_id
    if not triggered or not any(n_clicks_list):
        raise PreventUpdate
    key = triggered["key"]
    stem = Path(key).stem.lower()
    raw_dir = _session_dir(session_id) / "inputs" / "_raw"
    for p in raw_dir.glob("*"):
        if p.name == key or (p.stem.lower() == stem and p.suffix.lower() in VECTOR_SIDECAR_EXT):
            p.unlink(missing_ok=True)
    return (version or 0) + 1


@app.callback(
    Output("raw-version", "data", allow_duplicate=True),
    Output("role-overrides", "data", allow_duplicate=True),
    Input("clear-basket-btn", "n_clicks"),
    State("session-id", "data"),
    State("raw-version", "data"),
    prevent_initial_call=True,
)
def on_clear_basket(n_clicks, session_id, version):
    if not n_clicks:
        raise PreventUpdate
    raw_dir = _session_dir(session_id) / "inputs" / "_raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    return (version or 0) + 1, {}


@app.callback(
    Output("role-overrides", "data"),
    Input({"type": "role-select", "key": ALL}, "value"),
    State({"type": "role-select", "key": ALL}, "id"),
    State("role-overrides", "data"),
    prevent_initial_call=True,
)
def on_role_change(values, ids, overrides):
    overrides = dict(overrides or {})
    for value, id_ in zip(values, ids):
        overrides[id_["key"]] = (value or None)
    return overrides


@app.callback(
    Output("basket-items", "data"),
    Input("raw-version", "data"),
    Input("role-overrides", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def rebuild_basket_items(version, overrides, session_id):
    raw_dir = _session_dir(session_id) / "inputs" / "_raw"
    return _scan_and_classify(raw_dir, overrides or {})


@app.callback(
    Output("basket-list", "children"),
    Output("missing-warning", "children"),
    Output("run-button", "disabled"),
    Input("basket-items", "data"),
)
def render_basket(items):
    items = items or []
    list_children = [item_card(it) for it in items] if items else html.Div("No files added yet.", className="basket-empty")

    roles_present = [it["role"] for it in items if it["role"]]
    missing = [ROLE_LABELS[r] for r in REQUIRED_ROLES if r not in roles_present]
    dup_roles = {r for r in roles_present if roles_present.count(r) > 1}

    warnings = []
    if missing:
        warnings.append(html.Div(f"Missing required input(s): {', '.join(missing)}.", className="run-error"))
    if dup_roles:
        warnings.append(html.Div(
            "Multiple files assigned to: " + ", ".join(ROLE_LABELS[r] for r in dup_roles) + " — only the last one listed will be used.",
            className="run-warning",
        ))

    return list_children, warnings, bool(missing)


@app.callback(
    Output("results-container", "children"),
    Output("run-error", "children"),
    Input("run-button", "n_clicks"),
    State("basket-items", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_run(n_clicks, items, session_id):
    items = items or []
    role_paths = {it["role"]: it["paths"] for it in items if it["role"]}  # last match wins on duplicates

    missing = [ROLE_LABELS[r] for r in REQUIRED_ROLES if r not in role_paths]
    if missing:
        return dash.no_update, html.Div(f"Missing required input(s): {', '.join(missing)}.", className="run-error")

    session = _session_dir(session_id)
    results_dir = session / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)

    try:
        result = compute_canopy_cover(
            orthophoto_path=role_paths["ortho"][0],
            dsm_path=role_paths["dsm"][0],
            dtm_path=role_paths["dtm"][0],
            boundary_path=_pick_vector_path(role_paths["boundary"]),
            planting_rows_path=_pick_vector_path(role_paths["rows"]) if "rows" in role_paths else None,
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
