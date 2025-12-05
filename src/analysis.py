import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import hdbscan
from pathlib import Path


# =========================================================
# Load extracted embeddings
# =========================================================

features = np.load("features.npy")        # shape (N, 512)
ids = np.load("ids.npy")                  # shape (N,)

print("Loaded features:", features.shape)
print("Loaded IDs:", ids.shape)


# =========================================================
# PCA (dimensionality reduction before clustering)
# =========================================================

pca = PCA(n_components=50)
reduced = pca.fit_transform(features)

explained = pca.explained_variance_ratio_.sum()
print(f"PCA explained variance (50 comps): {explained:.4f}")

# Plot cumulative variance
plt.figure(figsize=(8,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Variance Explained")
plt.grid(True)
plt.savefig("pca_variance.png")
plt.close()


# =========================================================
# HDBSCAN clustering
# =========================================================

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=50,
    min_samples=10,
    metric='euclidean'
)

cluster_labels = clusterer.fit_predict(reduced)

unique = set(cluster_labels)
num_clusters = len(unique) - (1 if -1 in unique else 0)
noise_points = np.sum(cluster_labels == -1)

print(f"Clusters found: {num_clusters}")
print(f"Noise points: {noise_points}")

# Save cluster labels
np.save("cluster_labels.npy", cluster_labels)


# =========================================================
# UMAP (for manifold visualisation)
# =========================================================

umap_model = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean'
)

umap_embedding = umap_model.fit_transform(features)
np.save("umap_embedding.npy", umap_embedding)

# Plot UMAP embedding
plt.figure(figsize=(8,8))
plt.scatter(
    umap_embedding[:,0],
    umap_embedding[:,1],
    s=3,
    c=cluster_labels,
    cmap='Spectral'
)
plt.title("UMAP Embedding Colored by HDBSCAN Clusters")
plt.savefig("umap_clusters.png")
plt.close()


# =========================================================
# Cluster statistics output
# =========================================================

print("\nCluster sizes:")
unique_vals, counts = np.unique(cluster_labels, return_counts=True)
for label, count in zip(unique_vals, counts):
    print(f"Cluster {label}: {count} galaxies")

print("\nAnalysis complete. Outputs saved:")
print(" - pca_variance.png")
print(" - umap_clusters.png")
print(" - cluster_labels.npy")
print(" - umap_embedding.npy")
