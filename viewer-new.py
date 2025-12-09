import argparse
import base64
import io
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image

from dash import Dash, dcc, html, Input, Output, State, callback, no_update
import plotly.graph_objects as go


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

MAX_POINTS = 15000

MORPH_MAP = {
    "smooth":     "t01_smooth_or_features_a01_smooth_fraction",
    "disk":       "t01_smooth_or_features_a02_features_or_disk_fraction",
    "spiral":     "t04_spiral_a08_spiral_fraction",
    "bar":        "t03_bar_a06_bar_fraction",
    "merger":     "t08_odd_feature_a24_merger_fraction",
    "disturbed":  "t08_odd_feature_a21_disturbed_fraction",
    "ring":       "t08_odd_feature_a19_ring_fraction",
    "irregular":  "t08_odd_feature_a22_irregular_fraction",
    "bulge":      "t05_bulge_prominence_a12_obvious_fraction",
}


# ------------------------------------------------------------
# ARGPARSE
# ------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True, help="Folder under outputs/")
args = parser.parse_args()

TAG = args.tag
TAG_DIR = Path("outputs") / TAG


# ------------------------------------------------------------
# LOAD EMBEDDING + IDS
# ------------------------------------------------------------

umap = np.load(TAG_DIR / "umap_embedding.npy")
asset_ids = np.load("ids.npy").astype(int)

CLUSTER_FILES = {
    "merged": TAG_DIR / "merged_clusters.npy",
    "umap": TAG_DIR / "umap_clusters.npy",
    "pca": TAG_DIR / "pca_clusters.npy",
}


# ------------------------------------------------------------
# IMAGE DIRECTORY
# ------------------------------------------------------------

IMAGE_DIR = TAG_DIR / "images"
if not IMAGE_DIR.exists():
    IMAGE_DIR = Path("data/images")
print("using images from:", IMAGE_DIR)


# ------------------------------------------------------------
# LOAD METADATA
# ------------------------------------------------------------

maps = pd.read_csv("data/gz2maps.csv")
spec = pd.read_csv("data/gz2spec.csv")

maps["asset_id"] = pd.to_numeric(maps["asset_id"], errors="coerce").astype("Int64")
spec["dr7objid"] = pd.to_numeric(spec["dr7objid"], errors="coerce").astype("Int64")

df_base = pd.DataFrame({"idx": np.arange(len(asset_ids)), "asset_id": asset_ids})
df_base = df_base.merge(maps[["asset_id", "objid"]], on="asset_id", how="left")
df_base = df_base.rename(columns={"objid": "dr7objid"})
df_base = df_base.merge(spec, on="dr7objid", how="inner").reset_index(drop=True)

if len(df_base) > MAX_POINTS:
    df_base = df_base.sample(MAX_POINTS, random_state=42).reset_index(drop=True)

df_base["x"] = umap[df_base["idx"], 0]
df_base["y"] = umap[df_base["idx"], 1]

# morphology columns
for simple, raw in MORPH_MAP.items():
    df_base[simple] = df_base.get(raw, np.nan).astype(float)

# ------------------------------------------------------------
# DENSITY (fast 2D bin-count based estimate)
# ------------------------------------------------------------
# compute a sensible bin count based on sample size
_bins = int(min(150, max(40, int(np.sqrt(len(df_base)) * 3))))

_counts, xedges, yedges = np.histogram2d(df_base["x"].values, df_base["y"].values, bins=_bins)
# map each point to its bin index
_xi = np.clip(np.searchsorted(xedges, df_base["x"].values) - 1, 0, _counts.shape[0] - 1)
_yi = np.clip(np.searchsorted(yedges, df_base["y"].values) - 1, 0, _counts.shape[1] - 1)
_density = _counts[_xi, _yi].astype(float)

# add continuous and log-scaled density for coloring, plus 3-level categorical density
df_base["density"] = _density
df_base["density_log"] = np.log1p(_density)  # better dynamic range for color maps

_q33, _q66 = np.quantile(_density, [0.33, 0.66])
def _density_level(v):
    if v <= _q33:
        return "low"
    if v <= _q66:
        return "medium"
    return "high"
df_base["density_level"] = df_base["density"].apply(_density_level).astype("category")


# ------------------------------------------------------------
# IMAGE HELPERS (CACHED)
# ------------------------------------------------------------

@lru_cache(maxsize=100000)
def load_thumb(aid):
    """Return cached 128×128 PNG base64 thumbnail."""
    size = 128
    path = IMAGE_DIR / f"{aid}.jpg"
    if not path.exists():
        img = Image.new("RGB", (size, size), (40, 40, 40))
    else:
        img = Image.open(path).convert("RGB").resize((size, size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@lru_cache(maxsize=20000)
def load_full(aid):
    """Return cached full-res PNG base64 image."""
    path = IMAGE_DIR / f"{aid}.jpg"
    if not path.exists():
        img = Image.new("RGB", (256, 256))
    else:
        img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# pre-cache thumbs for scatter plot hover
df_base["thumb"] = df_base["asset_id"].apply(load_thumb)


# ------------------------------------------------------------
# FIGURE BUILDER
# ------------------------------------------------------------

def make_figure(df, mode="cluster"):
    # Improved colouring: make noise points (label == -1) gray and desaturated,
    # make clusters use a qualitative palette and add a thin outline so points stand out.
    fig = go.Figure()

    if mode == "cluster":
        labels = df["cluster"].to_numpy()

        # prepare qualitative palette from plotly.express
        try:
            import plotly.express as px
            palette = px.colors.qualitative.Dark24
        except Exception:
            # fallback simple palette
            palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                       "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

        unique_labels = sorted(set(labels))
        cluster_labels = [l for l in unique_labels if l != -1]

        # map real clusters to palette entries
        cmap = {lab: palette[i % len(palette)] for i, lab in enumerate(cluster_labels)}
        noise_color = "#bdbdbd"

        # Build cluster and noise masks
        mask_noise = labels == -1
        mask_cluster = ~mask_noise

        # Cluster marker (prominent)
        cluster_colors = [cmap.get(lab, palette[0]) for lab in labels[mask_cluster]]
        cluster_marker = dict(
            size=8,
            color=cluster_colors,
            opacity=0.98,
            line=dict(width=0.45, color="black")
        )

        # Noise marker (subtle gray)
        noise_marker = dict(
            size=5,
            color=noise_color,
            opacity=0.45,
            line=dict(width=0.15, color="#666666")
        )

        showscale = False

    elif mode == "density":  # continuous density
        marker = dict(
            size=6,
            color=df["density_log"].astype(float),
            colorscale="Viridis",
            showscale=True,
            opacity=0.85,
            line=dict(width=0.2, color="black")
        )

    elif mode == "density_level":  # discrete density levels
        # simple mapping for low/medium/high
        level_map = {"low": "#d4e6f1", "medium": "#5dade2", "high": "#1b4f72"}
        marker_colors = [level_map.get(lv, "#bdbdbd") for lv in df["density_level"].astype(str).values]
        marker = dict(size=7, color=marker_colors, opacity=0.9, line=dict(width=0.2, color="black"))

    else:
        # morphological continuous fields
        marker = dict(
            size=6,
            color=df[mode].astype(float),
            colorscale="Viridis",
            showscale=True,
            opacity=0.95,
            line=dict(width=0.2, color="black")
        )

    if mode == "cluster":
        # Draw noise first so clusters are rendered on top of it
        fig.add_trace(go.Scatter(
            x=df.loc[mask_noise, "x"],
            y=df.loc[mask_noise, "y"],
            mode="markers",
            marker=noise_marker,
            text=df.loc[mask_noise, "asset_id"].astype(str),
            customdata=np.stack([df.loc[mask_noise, "cluster"], df.loc[mask_noise, "thumb"], df.loc[mask_noise].get("density", np.zeros(mask_noise.sum()))], axis=1),
            hoverinfo="none",
            hovertemplate=None,
            name="noise",
        ))

        # Add clusters on top (prominent)
        fig.add_trace(go.Scatter(
            x=df.loc[mask_cluster, "x"],
            y=df.loc[mask_cluster, "y"],
            mode="markers",
            marker=cluster_marker,
            text=df.loc[mask_cluster, "asset_id"].astype(str),
            customdata=np.stack([df.loc[mask_cluster, "cluster"], df.loc[mask_cluster, "thumb"], df.loc[mask_cluster].get("density", np.zeros(mask_cluster.sum()))], axis=1),
            hoverinfo="none",
            hovertemplate=None,
            name="clusters",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers",
            marker=marker,
            text=df["asset_id"].astype(str),
            customdata=np.stack([df["cluster"], df["thumb"], df.get("density", np.zeros(len(df)))], axis=1),
            hoverinfo="none",
            hovertemplate=None,
        ))

    fig.update_layout(
        width=800,
        height=850,
        title=f"UMAP ({TAG})",
        dragmode="pan",
    )

    return fig


# ------------------------------------------------------------
# DASH APP
# ------------------------------------------------------------

app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.Label("Cluster source:"),
        dcc.Dropdown(
            id="cluster-source",
            options=[
                {"label": "PCA HDBSCAN clusters", "value": "pca"},
                {"label": "Merged clusters (recommended)", "value": "merged"},
                {"label": "UMAP HDBSCAN clusters", "value": "umap"},
            ],
            value="pca",
            style={"width": "350px"}
        ),

        html.Label("Colour mode:"),
        dcc.Dropdown(
            id="colour-mode",
            options=[{"label": "cluster", "value": "cluster"}] +
                    [{"label": k, "value": k} for k in MORPH_MAP.keys()] +
                    [{"label": "density (continuous)", "value": "density"},
                     {"label": "density level (low/medium/high)", "value": "density_level"}],
            value="cluster",
            clearable=False,
            style={"width": "300px", "margin-top": "10px"}
        ),

        dcc.Graph(id="scatter", clear_on_unhover=True),
        dcc.Tooltip(id="scatter-tooltip", direction="top"),
    ], style={"width": "65%", "display": "inline-block"}),

    html.Div([
        html.H3("Galaxy Preview"),
        html.Img(id="full-image", style={"width": "300px"}),
        html.Div(id="galaxy-info", style={"margin-top": "10px"}),

        html.Button("View Cluster Gallery", id="gallery-btn", n_clicks=0,
                    style={"margin-top": "20px"}),
        html.Div(id="gallery-output"),
    ], style={"width": "30%", "display": "inline-block",
              "vertical-align": "top", "padding": "20px"}),
])


# ------------------------------------------------------------
# CALLBACKS
# ------------------------------------------------------------

@callback(
    Output("scatter", "figure"),
    Input("cluster-source", "value"),
    Input("colour-mode", "value"),
)
def update_plot(cluster_source, colour_mode):
    df = df_base.copy()
    df["cluster"] = np.load(CLUSTER_FILES[cluster_source])[df["idx"]]
    return make_figure(df, mode=colour_mode)


# ---------- CURSOR TOOLTIP ----------
@callback(
    Output("scatter-tooltip", "show"),
    Output("scatter-tooltip", "bbox"),
    Output("scatter-tooltip", "children"),
    Input("scatter", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    pt = hoverData["points"][0]
    aid = int(pt["text"])
    bbox = pt["bbox"]

    thumb = load_thumb(aid)

    # try to fetch density info (safe guard)
    row = df_base[df_base["asset_id"] == aid]
    dens = None
    level = None
    if not row.empty:
        dens = float(row["density"].iloc[0])
        level = str(row["density_level"].iloc[0])

    extra = ""
    if dens is not None:
        extra = html.Div(f"Density: {dens:.0f} ({level})", style={"textAlign": "center", "margin-top": "4px"})

    tooltip = html.Div([
        html.Img(
            src=f"data:image/png;base64,{thumb}",
            style={"width": "400px", "display": "block", "margin": "0 auto"}
        ),
        html.Div(f"ID: {aid}", style={"textAlign": "center", "margin-top": "4px"}),
        extra
    ], style={"padding": "4px"})

    return True, bbox, tooltip


# ---------- CLICK FULL IMAGE ----------
@callback(
    Output("full-image", "src"),
    Output("galaxy-info", "children"),
    Input("scatter", "clickData"),
    Input("cluster-source", "value"),
)
def update_selected(click, cluster_source):
    if click is None:
        return None, "click a galaxy"

    aid = int(click["points"][0]["text"])
    df = df_base.copy()
    df["cluster"] = np.load(CLUSTER_FILES[cluster_source])[df["idx"]]
    row = df[df["asset_id"] == aid].iloc[0]

    # include density in the info string
    info = f"ID: {aid} | Cluster: {row['cluster']} | Density: {row.get('density', 0):.0f}"
    return f"data:image/png;base64,{load_full(aid)}", info


# ---------- GALLERY ----------
@callback(
    Output("gallery-output", "children"),
    Input("gallery-btn", "n_clicks"),
    State("scatter", "clickData"),
    State("cluster-source", "value"),
)
def show_gallery(n, click, cluster_source):
    if n == 0 or click is None:
        return ""

    aid = int(click["points"][0]["text"])
    df = df_base.copy()
    df["cluster"] = np.load(CLUSTER_FILES[cluster_source])[df["idx"]]

    clust = df[df["asset_id"] == aid]["cluster"].iloc[0]
    members = df[df["cluster"] == clust]

    thumbs = [
        html.Div([
            html.Img(src=f"data:image/png;base64,{row.thumb}"),
            html.Div(str(row.asset_id))
        ], style={"display": "inline-block", "margin": "5px"})
        for _, row in members.iterrows()
    ]

    return html.Div([
        html.H4(f"Cluster {clust} — {len(members)} galaxies"),
        html.Div(thumbs, style={"max-height": "500px", "overflow": "scroll"})
    ])


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------

if __name__ == "__main__":
    print("running at http://127.0.0.1:8050")
    app.run(debug=False)
