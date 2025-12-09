import numpy as np

um = np.load("outputs/final5/umap_embedding.npy")
pc = np.load("outputs/final5/pca_clusters.npy")
uc = np.load("outputs/final5/umap_clusters.npy")
mc = np.load("outputs/final5/merged_clusters.npy")
ids = np.load("ids.npy")

print(len(um), len(pc), len(uc), len(mc), len(ids))
