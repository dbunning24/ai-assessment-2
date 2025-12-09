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

umap_emb = np.load(TAG_DIR / "umap_embedding.npy")   # full, unfiltered, N rows
asset_ids = np.load("ids.npy").astype(int)           # N rows

assert umap_emb.shape[0] == len(asset_ids), "Mismatch between ids.npy and UMAP embedding"


# cluster arrays are 1:1 with features → ALSO N rows
CLUSTER_FILES = {
    "merged": TAG_DIR / "merged_clusters.npy",
    "umap":   TAG_DIR / "umap_clusters.npy",
    "pca":    TAG_DIR / "pca_clusters.npy",
}


# ------------------------------------------------------------
# IMAGE DIRECTORY
# ------------------------------------------------------------

IMAGE_DIR = TAG_DIR / "images"
if not IMAGE_DIR.exists():
    IMAGE_DIR = Path("data/images")


# ------------------------------------------------------------
# METADATA (NO ROW DROPS ALLOWED)
# ------------------------------------------------------------

maps = pd.read_csv("data/gz2maps.csv")
spec = pd.read_csv("data/gz2spec.csv")

maps["asset_id"] = pd.to_numeric(maps["asset_id"], errors="coerce")
spec["dr7objid"] = pd.to_numeric(spec["dr7objid"], errors="coerce")

# base dataframe ALWAYS contains ALL galaxies
df_base = pd.DataFrame({
    "asset_id": asset_ids,
    "real_idx": np.arange(len(asset_ids))  # gold-standard index match
})

# left joins → never drop rows
df_base = df_base.merge(maps[["asset_id", "objid"]], on="asset_id", how="left")
df_base = df_base.rename(columns={"objid": "dr7objid"})
df_base = df_base.merge(spec, on="dr7objid", how="left")

# attach UMAP coordinates
df_base["x"] = umap_emb[:, 0]
df_base["y"] = umap_emb[:, 1]

# morphology values (NaN if missing)
for simple, raw in MORPH_MAP.items():
    df_base[simple] = df_base.get(raw, np.nan).astype(float)


# ------------------------------------------------------------
# SUBSAMPLING (SAFE)
# ------------------------------------------------------------

if len(df_base) > MAX_POINTS:
    df_base = df_base.sample(MAX_POINTS, random_state=42).reset_index(drop=True)


# ------------------------------------------------------------
# DENSITY ESTIMATION
# ------------------------------------------------------------

_bins = int(min(150, max(40, int(np.sqrt(len(df_base)) * 3))))

_counts, xedges, yedges = np.histogram2d(
    df_base["x"], df_base["y"], bins=_bins
)
_xi = np.clip(np.searchsorted(xedges, df_base["x"]) - 1, 0, _counts.shape[0] - 1)
_yi = np.clip(np.searchsorted(yedges, df_base["y"]) - 1, 0, _counts.shape[1] - 1)

density_vals = _counts[_xi, _yi]
df_base["density"] = density_vals
df_base["density_log"] = np.log1p(density_vals)

_q33, _q66 = np.quantile(density_vals, [0.33, 0.66])
df_base["density_level"] = pd.cut(
    density_vals,
    bins=[-1, _q33, _q66, density_vals.max()+1],
    labels=["low", "medium", "high"]
)


# ------------------------------------------------------------
# IMAGE CACHING
# ------------------------------------------------------------

@lru_cache(maxsize=100000)
def load_thumb(aid):
    path = IMAGE_DIR / f"{aid}.jpg"
    img = Image.open(path).convert("RGB").resize((128, 128)) if path.exists() \
          else Image.new("RGB", (128,128), (40,40,40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@lru_cache(maxsize=20000)
def load_full(aid):
    path = IMAGE_DIR / f"{aid}.jpg"
    img = Image.open(path).convert("RGB") if path.exists() \
          else Image.new("RGB", (256,256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


df_base["thumb"] = df_base["asset_id"].apply(load_thumb)


# ------------------------------------------------------------
# FIGURE BUILDER (SINGLE TRACE, BULLETPROOF)
# ------------------------------------------------------------

def make_figure(df, mode="cluster"):

    # load cluster labels per selected source
    clusters_all = np.load(CLUSTER_FILES[current_cluster_source])
    clusters = clusters_all[df["real_idx"]]

    df = df.copy()
    df["cluster"] = clusters

    # ALWAYS embed row index → no more hover mismatch
    customdata = np.stack([
        df.index.values,       # index back into df_base
        df["asset_id"].values, # explicit redundancy
        df["cluster"].values,
        df["thumb"].values,
        df["density"].values,
    ], axis=1)

    if mode == "cluster":
        marker = dict(
            size=6,
            color=df["cluster"],
            colorscale="Spectral",
            showscale=True,
            opacity=0.9,
            line=dict(width=0.3, color="black"),
        )
        fig = go.Figure(go.Scatter(
            x=df["x"], y=df["y"],
            mode="markers",
            marker=marker,
            customdata=customdata,
            hoverinfo="none",
        ))

    elif mode == "density":
        marker = dict(
            size=6,
            color=df["density_log"],
            colorscale="Viridis",
            showscale=True,
            opacity=0.9,
        )
        fig = go.Figure(go.Scatter(
            x=df["x"], y=df["y"],
            mode="markers",
            marker=marker,
            customdata=customdata,
            hoverinfo="none",
        ))

    elif mode == "density_level":
        level_map = {"low": "#a6bddb", "medium": "#3690c0", "high": "#034e7b"}
        marker_colors = [level_map[str(v)] for v in df["density_level"]]
        fig = go.Figure(go.Scatter(
            x=df["x"], y=df["y"],
            mode="markers",
            marker=dict(size=6, color=marker_colors, opacity=0.95),
            customdata=customdata,
            hoverinfo="none",
        ))

    elif mode.startswith("heatmap_"):
        morph = mode.replace("heatmap_", "")
        bins = 120
        heat, X, Y = np.histogram2d(df["x"], df["y"], bins=bins, weights=df[morph])
        cnt, _, _ = np.histogram2d(df["x"], df["y"], bins=bins)
        avg = np.divide(heat, cnt, out=np.zeros_like(heat), where=cnt>0)

        fig = go.Figure(go.Heatmap(
            z=avg.T,
            x=X, y=Y,
            colorscale="Viridis",
            colorbar=dict(title=f"{morph} vote fraction"),
        ))

        # invisible scatter to enable clicking
        fig.add_trace(go.Scatter(
            x=df["x"], y=df["y"],
            mode="markers",
            marker=dict(size=2, color="rgba(0,0,0,0)"),
            customdata=customdata,
            hoverinfo="none",
        ))

    else:  # morphology continuous value
        marker = dict(
            size=6,
            color=df[mode].astype(float),
            colorscale="Viridis",
            showscale=True,
            opacity=0.95,
        )
        fig = go.Figure(go.Scatter(
            x=df["x"], y=df["y"],
            mode="markers",
            marker=marker,
            customdata=customdata,
            hoverinfo="none",
        ))

    fig.update_layout(
        width=800, height=850,
        dragmode="pan",
        title=f"UMAP — {mode}",
    )

    return fig


# ------------------------------------------------------------
# DASH APP
# ------------------------------------------------------------

app = Dash(__name__)
current_cluster_source = "merged"  # updated dynamically


app.layout = html.Div([
    html.Div([
        html.Label("Cluster source:"),
        dcc.Dropdown(
            id="cluster-source",
            options=[
                {"label": "PCA clusters", "value": "pca"},
                {"label": "UMAP clusters", "value": "umap"},
                {"label": "Merged clusters (recommended)", "value": "merged"},
            ],
            value="merged",
            style={"width": "300px"}
        ),

        html.Label("Colour mode:"),
        dcc.Dropdown(
            id="colour-mode",
            options=[{"label": "cluster", "value": "cluster"}] +
                    [{"label": k, "value": k} for k in MORPH_MAP.keys()] +
                    [{"label": f"heatmap: {k}", "value": f"heatmap_{k}"} for k in MORPH_MAP.keys()] +
                    [{"label": "density", "value": "density"},
                     {"label": "density level", "value": "density_level"}],
            value="cluster",
            style={"width": "300px", "margin-top": "10px"}
        ),

        dcc.Graph(id="scatter", clear_on_unhover=True),
        dcc.Tooltip(id="scatter-tooltip"),
    ], style={"display": "inline-block", "width": "65%"}),

    html.Div([
        html.H3("Galaxy Preview"),
        html.Img(id="full-image", style={"width": "300px"}),
        html.Div(id="galaxy-info"),
        html.Button("View Cluster Gallery", id="gallery-btn", n_clicks=0),
        html.Div(id="gallery-output"),
    ], style={"display": "inline-block", "width": "30%", "vertical-align": "top"}),
])


# ------------------------------------------------------------
# CALLBACKS
# ------------------------------------------------------------

@callback(
    Output("scatter", "figure"),
    Input("cluster-source", "value"),
    Input("colour-mode", "value"),
)
def update_plot(src, mode):
    global current_cluster_source
    current_cluster_source = src
    return make_figure(df_base, mode)


@callback(
    Output("scatter-tooltip", "show"),
    Output("scatter-tooltip", "bbox"),
    Output("scatter-tooltip", "children"),
    Input("scatter", "hoverData"),
)
def show_tooltip(hov):
    if hov is None:
        return False, no_update, no_update

    pt = hov["points"][0]
    row_idx = int(pt["customdata"][0])

    bbox = pt["bbox"]

    row = df_base.iloc[row_idx]
    thumb = row["thumb"]

    # compute cluster from current cluster source
    clusters_all = np.load(CLUSTER_FILES[current_cluster_source])
    cluster_val = clusters_all[row.real_idx]

    return True, bbox, html.Div([
        html.Img(src=f"data:image/png;base64,{row.thumb}", style={"width": "200px"}),
        html.Div(f"ID: {row.asset_id}"),
        html.Div(f"Cluster: {cluster_val}"),
        html.Div(f"Density: {row.density:.1f}"),
    ], style={"padding": "4px"})


@callback(
    Output("full-image", "src"),
    Output("galaxy-info", "children"),
    Input("scatter", "clickData"),
)
def show_selected(click):
    if click is None:
        return None, "Click a point"

    pt = click["points"][0]
    row_idx = int(pt["customdata"][0])
    row = df_base.iloc[row_idx]
    clusters_all = np.load(CLUSTER_FILES[current_cluster_source])
    cluster_val = clusters_all[row.real_idx]

    info = f"ID {row.asset_id} | Cluster {cluster_val}"
    img = load_full(row.asset_id)

    return f"data:image/png;base64,{img}", info


@callback(
    Output("gallery-output", "children"),
    Input("gallery-btn", "n_clicks"),
    State("scatter", "clickData"),
)
def show_gallery(n, click):
    if n == 0 or click is None:
        return ""

    pt = click["points"][0]
    row_idx = int(pt["customdata"][0])
    row = df_base.iloc[row_idx]

    # compute cluster for this row
    clusters_all = np.load(CLUSTER_FILES[current_cluster_source])
    cluster_val = clusters_all[row.real_idx]

    # build df with cluster column attached
    df_temp = df_base.copy()
    df_temp["cluster"] = clusters_all[df_temp["real_idx"]]

    # find members of the same cluster
    members = df_temp[df_temp["cluster"] == cluster_val]

    thumbs = [
        html.Div([
            html.Img(src=f"data:image/png;base64,{m.thumb}", style={"width": "120px"}),
            html.Div(str(m.asset_id))
        ], style={"display": "inline-block", "margin": "5px"})
        for _, m in members.iterrows()
    ]

    return html.Div([
        html.H4(f"Cluster {cluster_val} — {len(members)} galaxies"),
        html.Div(thumbs, style={"max-height": "500px", "overflow": "scroll"})
    ])

# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------

if __name__ == "__main__":
    print("Running at http://127.0.0.1:8050")
    app.run(debug=False)
