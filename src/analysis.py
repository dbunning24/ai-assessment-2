import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import hdbscan
from pathlib import Path
import argparse
import pandas as pd


# ------------------------------------------------------------
# Morphology vote-fraction mapping
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
# entropy helper
# ------------------------------------------------------------
def entropy(p):
    p = np.clip(p, 1e-9, 1)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


# ------------------------------------------------------------
# noise reclustering using HDBSCAN again
# ------------------------------------------------------------
def recluster_noise(features, labels):
    noise_idx = np.where(labels == -1)[0]
    if len(noise_idx) == 0:
        return labels

    noise_features = features[noise_idx]

    secondary = hdbscan.HDBSCAN(
        min_cluster_size=8,
        min_samples=2,
        cluster_selection_method="leaf"
    ).fit_predict(noise_features)

    # Shift secondary cluster IDs so they don't overlap main clusters
    max_label = labels.max()
    secondary = np.where(secondary == -1, -1, secondary + max_label + 1)

    new_labels = labels.copy()
    new_labels[noise_idx] = secondary

    return new_labels


# ------------------------------------------------------------
# MAIN ANALYSIS
# ------------------------------------------------------------
def run_analysis(tag: str, n_pca: int, min_cluster_size: int, min_samples: int):

    outdir = Path("outputs") / tag
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Running analysis with tag: {tag} ===\n")

    # -----------------------------------------------------
    # Load features + IDs
    # -----------------------------------------------------
    features = np.load("features.npy")
    ids = np.load("ids.npy").astype(int)

    print("Loaded features:", features.shape)
    print("Loaded IDs:", ids.shape)

    # -----------------------------------------------------
    # Load vote fraction metadata
    # -----------------------------------------------------
    maps = pd.read_csv("data/gz2maps.csv")
    spec = pd.read_csv("data/gz2spec.csv")

    maps["asset_id"] = pd.to_numeric(maps["asset_id"], errors="coerce").astype("Int64")
    spec["dr7objid"] = pd.to_numeric(spec["dr7objid"], errors="coerce").astype("Int64")

    df = pd.DataFrame({"asset_id": ids})
    df = df.merge(maps[["asset_id", "objid"]], on="asset_id", how="left")
    df = df.rename(columns={"objid": "dr7objid"})
    df = df.merge(spec, on="dr7objid", how="inner")

    print("Aligned vote fraction rows:", len(df))


    # -----------------------------------------------------
    # PCA
    # -----------------------------------------------------
    pca = PCA(n_components=n_pca)
    reduced = pca.fit_transform(features)

    total_var = pca.explained_variance_ratio_.sum()
    print(f"PCA explained variance: {total_var:.4f}")

    plt.figure(figsize=(8,4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title(f"PCA Variance Explained ({tag})")
    plt.savefig(outdir / "pca_variance.png")
    plt.close()

    # -----------------------------------------------------
    # HDBSCAN on PCA (main scientific clustering)
    # -----------------------------------------------------
    print("\nRunning HDBSCAN on PCA...")
    clusterer_pca = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="leaf",
        cluster_selection_epsilon=0.01,
    )

    pca_labels = clusterer_pca.fit_predict(reduced)
    print("PCA clusters:", len(set(pca_labels)) - (1 if -1 in pca_labels else 0))
    print("PCA noise:", np.sum(pca_labels == -1))

    np.save(outdir / "pca_clusters.npy", pca_labels)


    # -----------------------------------------------------
    # UMAP embedding for visualization + alternative clustering
    # -----------------------------------------------------
    print("\nRunning UMAP...")
    umap_model = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        metric="euclidean"
    )
    umap_emb = umap_model.fit_transform(features)

    from sklearn.manifold import trustworthiness

    tw = trustworthiness(features, umap_emb, n_neighbors=30)
    print("UMAP Trustworthiness:", tw)
    df["umap_x"] = umap_emb[:,0]
    df["umap_y"] = umap_emb[:,1]

    np.save(outdir / "umap_embedding.npy", umap_emb)

    # Plot PCA clusters on UMAP
    plt.figure(figsize=(8,8))
    plt.scatter(umap_emb[:,0], umap_emb[:,1], s=3, c=pca_labels, cmap="Spectral")
    plt.title("UMAP coloured by PCA-HDBSCAN clusters")
    plt.savefig(outdir / "umap_clusters.png")
    plt.close()


    # -----------------------------------------------------
    # HDBSCAN directly on UMAP (better separation, less noise)
    # -----------------------------------------------------
    print("\nRunning HDBSCAN on UMAP...")
    clusterer_umap = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=3,
        cluster_selection_method="leaf"
    )
    umap_labels = clusterer_umap.fit_predict(umap_emb)
    print("UMAP clusters:", len(set(umap_labels)) - (1 if -1 in umap_labels else 0))
    print("UMAP noise:", np.sum(umap_labels == -1))

    np.save(outdir / "umap_clusters.npy", umap_labels)


    # -----------------------------------------------------
    # Noise reduction: recluster UMAP noise
    # -----------------------------------------------------
    print("\nReclustering UMAP noise...")
    umap_labels_final = recluster_noise(umap_emb, umap_labels)
    np.save(outdir / "merged_clusters.npy", umap_labels_final)

    print("Final clusters:", len(set(umap_labels_final)) - (1 if -1 in umap_labels_final else 0))
    print("Final noise:", np.sum(umap_labels_final == -1))


    # attach final labels
    df["cluster"] = umap_labels_final


    # -----------------------------------------------------
    # Save analysis CSV
    # -----------------------------------------------------
    print("\nSaving analysis CSV...")
    for simple, raw in MORPH_MAP.items():
        df[simple] = df[raw].astype(float)

    df.to_csv(outdir / "analysis.csv", index=False)
    print("Saved analysis.csv")

    # cluster statistics
    print("Saving cluster summary...")
    cluster_summary = (
        df.groupby("cluster")[list(MORPH_MAP.keys())]
        .agg(["mean","std","min","max","count"])
    )
    cluster_summary.to_csv(outdir / "cluster_summary.csv")

    print("\nAll done.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PCA + UMAP + HDBSCAN analysis.")

    parser.add_argument("--tag", type=str, default="default",
                        help="Output folder name under outputs/")
    parser.add_argument("--n-pca", type=int, default=20,
                        help="Number of PCA components")
    parser.add_argument("--min-cluster-size", type=int, default=20,
                        help="HDBSCAN min_cluster_size (PCA mode)")
    parser.add_argument("--min-samples", type=int, default=5,
                        help="HDBSCAN min_samples (PCA mode)")

    args = parser.parse_args()

    run_analysis(
        tag=args.tag,
        n_pca=args.n_pca,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )
