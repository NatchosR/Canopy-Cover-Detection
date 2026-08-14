"""Canopy cover detection -- Web UI.

Browse precomputed canopy KPIs per plot, inspect the mature/young/other
vegetation layers over the orthophoto, and download all outputs (GeoTIFF
mask, GeoPackage layers, GeoJSON, summary JSON).

Run with: venv/Scripts/python app.py
"""
import json
from pathlib import Path

import dash
import flask
from dash import Dash, Input, Output, State, dcc, html

OUTPUTS_DIR = Path("outputs")

DOWNLOADABLE_FILES = [
    ("canopy_mask.tif", "Canopy mask (GeoTIFF, 4 bands)"),
    ("canopy_layers.gpkg", "Canopy layers (GeoPackage)"),
    ("canopy_polygons.geojson", "Canopy polygons (GeoJSON)"),
    ("summary.json", "KPI summary (JSON)"),
    ("overlay.png", "Overlay preview (PNG)"),
]
WEB_ASSET_FILES = {"web_base_rgb.png", "web_mature_overlay.png", "web_young_overlay.png", "web_other_overlay.png"}
ALLOWED_FILES = {f for f, _ in DOWNLOADABLE_FILES} | WEB_ASSET_FILES


def discover_plots():
    if not OUTPUTS_DIR.exists():
        return []
    return sorted(d.name for d in OUTPUTS_DIR.iterdir() if d.is_dir() and (d / "summary.json").exists())


def load_summary(plot):
    with open(OUTPUTS_DIR / plot / "summary.json") as f:
        return json.load(f)


app = Dash(__name__, title="Canopy Cover Detection")
server = app.server


@server.route("/files/<plot>/<path:filename>")
def serve_file(plot, filename):
    if plot not in discover_plots() or filename not in ALLOWED_FILES:
        flask.abort(404)
    directory = (OUTPUTS_DIR / plot).resolve()
    if not (directory / filename).exists():
        flask.abort(404)
    return flask.send_from_directory(directory, filename)


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
        kpi_card(
            "KPI1 — Total Canopy",
            f"{summary['kpi1_total_canopy_pct']:.2f}%",
            f"{summary['kpi1_total_canopy_m2']:.0f} m² · mature + young",
            "#1b4332",
        ),
        kpi_card(
            "KPI2 — Mature Trees",
            f"{summary['kpi2_mature_canopy_pct']:.2f}%",
            f"{summary['n_mature_tree_blobs']} blobs",
            "#c1121f",
        ),
        kpi_card(
            "KPI3 — Young Tree Candidates",
            f"{summary['kpi3_young_canopy_pct']:.2f}%",
            f"{summary['n_young_tree_blobs']} blobs (potential trees)",
            "#0891a8",
        ),
        kpi_card(
            "Other Vegetation (mato)",
            f"{summary['other_vegetation_pct']:.2f}%",
            f"{summary['n_other_vegetation_blobs']} blobs — not a KPI",
            "#e07a00",
        ),
    ]


def build_validation_banner(summary):
    validation = summary.get("tree_location_validation")
    if validation:
        text = (
            f"Tree-location recall check: {validation['recall_pct']:.1f}% of {validation['n_known_trees']} tracked "
            f"trees matched a detected blob within {validation['match_radius_m']} m "
            f"({validation['n_known_matched_by_young_blob']} via young candidates, "
            f"{validation['n_known_matched_by_mature_blob']} via mature). "
            f"{validation['n_young_blobs_unmatched']} of {validation['n_young_blobs_total']} young blobs had no "
            "known tree nearby. Recall proxy only — the location file has correct sequence but inaccurate "
            "coordinates, this is not a precision measure."
        )
        return html.Div(text, className="validation-banner")
    return html.Div(
        [html.Strong("Validation status: "), summary.get("validation_status", "unvalidated")],
        className="validation-banner warn",
    )


def build_viewer(plot, layer_values):
    def vis(name):
        return "block" if name in layer_values else "none"

    return [
        html.Img(src=f"/files/{plot}/web_base_rgb.png", className="layer-img"),
        html.Img(
            src=f"/files/{plot}/web_mature_overlay.png",
            className="overlay-img",
            style={"display": vis("mature")},
        ),
        html.Img(
            src=f"/files/{plot}/web_young_overlay.png",
            className="overlay-img",
            style={"display": vis("young")},
        ),
        html.Img(
            src=f"/files/{plot}/web_other_overlay.png",
            className="overlay-img",
            style={"display": vis("other")},
        ),
    ]


def build_downloads(plot):
    return [
        html.A(label, href=f"/files/{plot}/{fname}", download=fname, className="download-link")
        for fname, label in DOWNLOADABLE_FILES
    ]


def build_layout():
    plots = discover_plots()
    default_plot = plots[0] if plots else None
    default_summary = load_summary(default_plot) if default_plot else {}
    default_layers = ["mature", "young", "other"]

    return html.Div(
        [
            html.Div(
                [
                    html.H1("Canopy Cover Detection"),
                    html.P(
                        "Agroforestry drone-survey canopy KPIs — provisional until validated against "
                        "field-verified reference points.",
                        className="subtitle",
                    ),
                ],
                className="header",
            ),
            html.Div(
                [
                    html.Label("Plot"),
                    dcc.Dropdown(
                        id="plot-dropdown",
                        options=[{"label": p, "value": p} for p in plots],
                        value=default_plot,
                        clearable=False,
                        style={"width": "260px"},
                    ),
                ],
                className="controls",
            ),
            html.Div(id="validation-banner", children=build_validation_banner(default_summary) if default_summary else None),
            html.Div(id="kpi-row", className="kpi-row", children=build_kpi_row(default_summary) if default_summary else None),
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
                            html.Div(id="download-links", children=build_downloads(default_plot) if default_plot else None),
                        ],
                        className="sidebar",
                    ),
                    html.Div(
                        html.Div(
                            id="image-viewer",
                            className="image-viewer",
                            children=build_viewer(default_plot, default_layers) if default_plot else None,
                        ),
                        className="viewer-container",
                    ),
                ],
                className="main-row",
            ),
            html.Details(
                [
                    html.Summary("Detection parameters used"),
                    html.Pre(id="config-json", children=json.dumps(default_summary.get("config", {}), indent=2)),
                ],
                className="config-panel",
            ),
            dcc.Store(id="current-plot", data=default_plot),
        ],
        className="app-container",
    )


app.layout = build_layout


@app.callback(
    Output("kpi-row", "children"),
    Output("validation-banner", "children"),
    Output("download-links", "children"),
    Output("config-json", "children"),
    Output("current-plot", "data"),
    Input("plot-dropdown", "value"),
)
def on_plot_change(plot):
    if not plot:
        return [], None, [], "{}", None
    summary = load_summary(plot)
    return (
        build_kpi_row(summary),
        build_validation_banner(summary),
        build_downloads(plot),
        json.dumps(summary.get("config", {}), indent=2),
        plot,
    )


@app.callback(
    Output("image-viewer", "children"),
    Input("current-plot", "data"),
    Input("layer-toggle", "value"),
)
def on_viewer_change(plot, layer_values):
    if not plot:
        return []
    return build_viewer(plot, layer_values or [])


if __name__ == "__main__":
    app.run(debug=True, port=8050)
