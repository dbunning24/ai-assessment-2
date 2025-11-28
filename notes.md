# reminder

i’m building an unsupervised machine-learning pipeline to cluster galaxies based purely on their visual appearance, using images from the galaxy zoo 2 dataset.

the system starts with simclr, a contrastive learning method.
simclr takes each galaxy image, makes an augmented version of it (random crops, flips, colour jitter, etc), and treats those two as a positive pair. everything else in the batch becomes implicitly negative, because simclr assumes they’re different galaxies. the model learns to pull positive pairs together in embedding space and push negatives apart, basically figuring out what “similar” galaxies look like without ever being told the correct labels.

this gives me high-dimensional embeddings, which i will apply pca to shrink them down while keeping the core structure of the data intact.

next comes hdbscan, a density-based clustering algorithm. instead of starting with random points, it builds a minimum spanning tree of the embeddings, forms a hierarchy of density regions, then extracts the most stable clusters. galaxies that don’t fit neatly into any dense region get labelled as noise.

for visualisation, i use umap on the embeddings to produce a 2d projection.
this gives me a bird’s-eye view of how the galaxies arrange themselves: tight groups, long arcs, and outliers.

finally, i manually inspect the results. i check how the umap plot looks, how the clusters from hdbscan line up, and whether these unsupervised groups correspond in any meaningful way to the galaxy zoo vote fractions — spirals together, ellipticals together, or weird mergers.

# todo

-   read relevant papers on unsupervised learning in astronomy and galaxy morphology
-   download gz2specz.csv from galaxy zoo 2 on to laptop
-   install separate pytorch versions for pc (cudo) and laptop (cpu only)
-   set up data pipeline to load images and preprocess them for simclr
-   implement simclr training loop
-   run simclr on galaxy zoo 2 images to get embeddings
-   apply pca to reduce embedding dimensionality
-   run hdbscan on reduced embeddings to find clusters
-   use umap to visualise embeddings in 2d
-   analyse clusters: compare to galaxy zoo vote fractions, visualise representative images from each cluster
-   iterate: tweak simclr augmentations, hdbscan parameters, etc to improve clustering if needed
-   write paper documenting methodology and findings
