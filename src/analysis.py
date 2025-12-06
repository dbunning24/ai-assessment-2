import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import hdbscan
from pathlib import Path
import argparse
import pandas as pd


# ------------------------------------------------------------
# Morphology vote-fraction mapping (same as viewer)
# ------------------------------------------------------------

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
# entropy helper for cluster purity
# ------------------------------------------------------------
def entropy(p):
    p = np.clip(p, 1e-9, 1)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def run_analysis(tag: str, n_pca: int, min_cluster_size: int, min_samples: int):

    # -----------------------------------------------------
    # Setup output directory
    # -----------------------------------------------------
    outdir = Path("outputs") / tag
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Running analysis with tag: {tag} ===")
    print(f"Outputs will be saved to: {outdir}\n")

    # -----------------------------------------------------
    # Load embeddings + IDs
    # -----------------------------------------------------
    features = np.load("features.npy")
    ids = np.load("ids.npy").astype(int)

    print("Loaded features:", features.shape)
    print("Loaded IDs:", ids.shape)


    # -----------------------------------------------------
    # Load morphology vote fractions (same merge as viewer)
    # -----------------------------------------------------
    print("\nLoading vote fraction data...")

    maps = pd.read_csv("data/gz2maps.csv")       # asset_id → objid
    spec = pd.read_csv("data/gz2spec.csv")       # dr7objid → vote fractions

    maps["asset_id"] = pd.to_numeric(maps["asset_id"], errors="coerce").astype("Int64")
    maps["objid"] = pd.to_numeric(maps["objid"], errors="coerce").astype("Int64")
    spec["dr7objid"] = pd.to_numeric(spec["dr7objid"], errors="coerce").astype("Int64")

    df = pd.DataFrame({"asset_id": ids})

    # merge asset_id → objid
    df = df.merge(maps[["asset_id", "objid"]], on="asset_id", how="left")

    # rename objid→dr7objid to match spec
    df = df.rename(columns={"objid": "dr7objid"})

    # merge vote fractions
    df = df.merge(spec, on="dr7objid", how="inner")

    print("Aligned vote fraction rows:", len(df))


    # -----------------------------------------------------
    # PCA dimensionality reduction
    # -----------------------------------------------------
    pca = PCA(n_components=n_pca)
    reduced = pca.fit_transform(features)

    total_var = pca.explained_variance_ratio_.sum()
    print(f"PCA explained variance ({n_pca} comps): {total_var:.4f}")

    # save plot
    plt.figure(figsize=(8,4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(f"PCA Variance Explained ({tag})")
    plt.grid(True)
    plt.savefig(outdir / "pca_variance.png")
    plt.close()


    # -----------------------------------------------------
    # HDBSCAN clustering
    # -----------------------------------------------------
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="leaf",
        cluster_selection_epsilon=0.01
    )

    cluster_labels = clusterer.fit_predict(reduced)
    df["cluster"] = cluster_labels

    unique = set(cluster_labels)
    num_clusters = len(unique) - (1 if -1 in unique else 0)
    noise_points = np.sum(cluster_labels == -1)

    print(f"Clusters found: {num_clusters}")
    print(f"Noise points: {noise_points}")

    np.save(outdir / "cluster_labels.npy", cluster_labels)


    # -----------------------------------------------------
    # UMAP embedding
    # -----------------------------------------------------
    umap_model = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        metric="euclidean"
    )

    umap_embedding = umap_model.fit_transform(features)
    df["umap_x"] = umap_embedding[:,0]
    df["umap_y"] = umap_embedding[:,1]

    np.save(outdir / "umap_embedding.npy", umap_embedding)

    # plot
    plt.figure(figsize=(8,8))
    plt.scatter(
        umap_embedding[:,0],
        umap_embedding[:,1],
        s=3,
        c=cluster_labels,
        cmap="Spectral"
    )
    plt.title(f"UMAP Embedding Colored by HDBSCAN Clusters ({tag})")
    plt.savefig(outdir / "umap_clusters.png")
    plt.close()


    # -----------------------------------------------------
    # Save per-galaxy analysis CSV
    # -----------------------------------------------------
    print("\nSaving per-galaxy analysis CSV...")

    analysis_cols = (
        ["asset_id", "cluster", "umap_x", "umap_y"] +
        list(MORPH_MAP.keys())
    )

    # rename morphology columns
    for simple, raw in MORPH_MAP.items():
        df[simple] = df[raw].astype(float)

    df[analysis_cols].to_csv(outdir / "analysis.csv", index=False)

    print(f"Wrote analysis.csv with {len(df)} rows.")


    # -----------------------------------------------------
    # Cluster summary statistics (mean/std/min/max/count)
    # -----------------------------------------------------
    print("Computing cluster summary statistics...")

    cluster_summary = (
        df.groupby("cluster")[list(MORPH_MAP.keys())]
        .agg(["mean", "std", "min", "max", "count"])
    )

    cluster_summary.to_csv(outdir / "cluster_summary.csv")
    print("Saved cluster_summary.csv")


    # -----------------------------------------------------
    # Entropy per cluster (purity measurement)
    # -----------------------------------------------------
    print("Computing entropy per cluster...")

    entropy_dict = {}

    for cluster_id, sub in df.groupby("cluster"):
        stats = {}
        for morph in MORPH_MAP.keys():
            p = sub[morph].mean()
            stats[morph + "_entropy"] = float(entropy(p))
        entropy_dict[cluster_id] = stats

    entropy_df = pd.DataFrame.from_dict(entropy_dict, orient="index")
    entropy_df.index.name = "cluster"
    entropy_df.to_csv(outdir / "cluster_entropy.csv")

    print("Saved cluster_entropy.csv")


    # -----------------------------------------------------
    # Final cluster size printout
    # -----------------------------------------------------
    print("\nCluster sizes:")
    unique_vals, counts = np.unique(cluster_labels, return_counts=True)
    for label, count in zip(unique_vals, counts):
        print(f"Cluster {label}: {count} galaxies")

    print(f"\nAnalysis complete. Outputs saved in outputs/{tag}/")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PCA + UMAP + HDBSCAN analysis.")

    parser.add_argument("--tag", type=str, default="default",
                        help="Name of output folder under outputs/")
    parser.add_argument("--n-pca", type=int, default=50,
                        help="Number of PCA components")
    parser.add_argument("--min-cluster-size", type=int, default=50,
                        help="HDBSCAN min_cluster_size")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="HDBSCAN min_samples")

    args = parser.parse_args()

    run_analysis(
        tag=args.tag,
        n_pca=args.n_pca,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )
