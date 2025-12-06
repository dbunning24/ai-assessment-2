import argparse
import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from dash import Dash, dcc, html, Input, Output, State, callback
import plotly.graph_objects as go


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

MAX_POINTS = 5000

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

# detect cluster files
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
# LOAD CSV METADATA
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
# IMAGE HELPERS
# ------------------------------------------------------------

def load_thumb(aid, size=128):
    path = IMAGE_DIR / f"{aid}.jpg"
    if not path.exists():
        img = Image.new("RGB", (size, size), (40, 40, 40))
    else:
        img = Image.open(path).convert("RGB").resize((size, size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def load_full(aid):
    path = IMAGE_DIR / f"{aid}.jpg"
    if not path.exists():
        img = Image.new("RGB", (256, 256))
    else:
        img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# cache thumbnails
df_base["thumb"] = df_base["asset_id"].apply(load_thumb)


# ------------------------------------------------------------
# FIGURE BUILDER
# ------------------------------------------------------------

def make_figure(df, mode="cluster"):
    if mode == "cluster":
        colors = df["cluster"].astype("category")
        colorscale = None
        showscale = False
    else:
        colors = df[mode].astype(float)
        colorscale = "Viridis"
        showscale = True

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["x"],
        y=df["y"],
        mode="markers",
        marker=dict(
            size=6,
            color=colors,
            colorscale=colorscale,
            showscale=showscale,
            opacity=0.85,
        ),
        text=df["asset_id"].astype(str),
        customdata=np.stack([df["cluster"], df["thumb"]], axis=1),
        hovertemplate=(
            "<b>ID:</b> %{text}<br>"
            "<b>Cluster:</b> %{customdata[0]}<br>"
            "<br><img src='data:image/png;base64,%{customdata[1]}' width='120'><br>"
        ),
    ))

    fig.update_layout(
        width=800,
        height=850,
        title=f"UMAP ({TAG})",
        dragmode="pan"
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
                {"label": "Merged clusters (recommended)", "value": "merged"},
                {"label": "UMAP HDBSCAN clusters", "value": "umap"},
                {"label": "PCA HDBSCAN clusters", "value": "pca"},
            ],
            value="merged",
            style={"width": "350px"}
        ),

        html.Label("Colour mode:"),
        dcc.Dropdown(
            id="colour-mode",
            options=[{"label": "cluster", "value": "cluster"}] +
                    [{"label": k, "value": k} for k in MORPH_MAP.keys()],
            value="cluster",
            clearable=False,
            style={"width": "300px", "margin-top": "10px"}
        ),

        dcc.Graph(id="scatter"),
    ], style={"width": "65%", "display": "inline-block"}),

    html.Div([
        html.H3("Galaxy Preview"),
        html.Img(id="full-image", style={"width": "300px"}),
        html.Div(id="galaxy-info", style={"margin-top": "10px"}),

        html.Button("View Cluster Gallery", id="gallery-btn", n_clicks=0,
                    style={"margin-top": "20px"}),
        html.Div(id="gallery-output"),
    ], style={"width": "30%", "display": "inline-block", "vertical-align": "top",
              "padding": "20px"}),
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
    info = f"ID: {aid} | Cluster: {row['cluster']}"

    return f"data:image/png;base64,{load_full(aid)}", info


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
            html.Img(
                src=f"data:image/png;base64,{row.thumb}"
            ),
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
