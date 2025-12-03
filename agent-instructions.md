Role

You are an expert Machine Learning Engineer specializing in Astrophysics and Computer Vision. Your task is to implement an unsupervised Deep Learning pipeline for Galaxy Morphology discovery using PyTorch and PyTorch Lightning.

Project Goal

We are replicating a research paper titled "Unsupervised Galaxy Morphology Clustering with SimCLR and HDBSCAN". The goal is to train a contrastive learning model (SimCLR) on unlabelled galaxy images to learn morphological features, and then validate these features by clustering them and comparing them to human vote fractions from the Galaxy Zoo 2 dataset.

Tech Stack

Language: Python 3.11

Framework: PyTorch (torch, torchvision), PyTorch Lightning (pytorch-lightning)

Data Handling: Pandas, NumPy, PIL

Clustering/Dim Reduction: scikit-learn (PCA), hdbscan, umap-learn

Visualization: Matplotlib, Seaborn, TensorBoard

Implementation Steps

0. Environment Configuration (environment.yml)

Action: Update the environment.yml file to ensure the correct dependencies are present for the pipeline.

CRITICAL CONSTRAINT: Do NOT change the environment name. It must remain name: galaxy-morphology.

Required Updates:

Channels: Ensure pytorch and conda-forge are listed before defaults.

Add: torchvision (Required for transforms).

Add: pytorch (Explicitly list alongside pytorch-lightning).

Remove: albumentations (We will use native torchvision transforms to ensure strict control over astrophysical augmentations).

Keep: hdbscan, umap-learn, tensorboard, pytorch-lightning, pandas, scikit-learn.

0.1. Hardware & Platform Optimizations

The code must run seamlessly on two distinct hardware profiles. Implement logic (via argparse or auto-detection) to switch between these configurations.

Crucial Note on SimCLR Performance:
SimCLR performance is directly tied to Batch Size (number of negative samples).

Profile A (Desktop) is the "Scientific Run" (High quality features).

Profile B (Laptop) is the "Development Run" (Lower quality features, used for debugging pipeline).

Learning Rate Scaling: To maintain training stability across widely different batch sizes, implement Linear Learning Rate Scaling: $LR = BaseLR \times (BatchSize / 256)$.

Configurations:

Profile A: Desktop PC (Windows 11 + NVIDIA GPU)

Accelerator: gpu (CUDA).

Precision: 16-mixed (Enables Automatic Mixed Precision).

Batch Size: 256 (Standard for SimCLR).

Workers: Default to 0 or 4 (Check for Windows "Broken Pipe" stability).

Profile B: Laptop (Arch Linux + Integrated GPU)

Accelerator: auto (CPU).

Precision: 32 (Standard float32).

Batch Size: 64 (Minimum viable for testing).

Workers: 4 (Safe on Linux).

1. Environment & Data Setup

Directory Structure:

data/gz2spec.csv

data/gz2maps.csv

data/images/ (Contains raw images).

Preprocessing Logic:

Load Data: Load gz2spec.csv.

Subset Selection: Randomly sample 100,000 galaxies (or top 100k by magnitude) to reduce computational load.

Filtering: Filter for z > 0 (extragalactic only).

Image Processing:

Central Crop (remove background).

Resize to 64x64 (preferred for batch size) or max 96x96.

Normalization: Calculate mean/std from a 10k sample subset.

2. The Custom Dataset (src/dataset.py)

Create GalaxyZooDataset(Dataset).

Output: __getitem__ must return (view1, view2, galaxy_id).

Augmentation (Use torchvision.transforms):

Note: Do not use Albumentations; stick to Torchvision for native Tensor integration.

Include:

RandomResizedCrop (scale=(0.5, 1.0))

RandomHorizontalFlip (p=0.5)

RandomVerticalFlip (p=0.5)

RandomRotation (90, 180, 270)

ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05) - Mild only.

Exclude: RandomGrayscale (colour is physical), GaussianBlur (blurs arms).

3. The Lightning Module (src/model.py)

Create SimCLR(pl.LightningModule).

Components:

self.encoder: ResNet18 (remove fc, adjust conv1 kernel if images < 96px).

self.projection_head: MLP (512 -> 128).

self.criterion: NT-Xent Loss.

Key Methods:

forward(x): Returns features (encoder output).

training_step(batch, batch_idx):

Unpack (view1, view2, _).

Pass both views through encoder + projection head.

Compute NT-Xent loss.

Log train_loss to TensorBoard.

configure_optimizers():

Apply Linear Scaling Rule: lr = 1e-3 * (batch_size / 256).

Return Adam optimizer with this scaled LR.

4. The Loss Function (src/loss.py)

Implement NT-Xent Loss.

Logic: Cosine similarity matrix between all views in batch. Maximize agreement between positive pairs (i, i+batch_size), minimize others. Temperature $\tau=0.5$.

5. Training (src/train.py)

Setup pl.Trainer dynamically based on Hardware Profile (Section 0.1).

Initialize SimCLR module.

Initialize DataLoader with profile-specific batch size and workers.

Run trainer.fit(model, dataloader).

6. Analysis Pipeline (notebooks/analysis.ipynb)

Feature Extraction:

Load model from checkpoint.

Run inference on the 100k subset using the Encoder only (discard projection head).

Save features X and galaxy_ids.

Methodology Steps:

PCA: Reduce to ~50 dims (95% variance).

HDBSCAN: Cluster the PCA output (min_cluster_size=50). Use the hdbscan library (not sklearn).

UMAP: Project original features to 2D for viz.

Validation:

Merge Cluster IDs and UMAP coords with gz2spec.csv.

Plot UMAP colored by Cluster ID.

Plot UMAP colored by Vote Fractions (e.g., smooth_fraction, features_fraction) to verify physical meaning.

Style Guidelines

Use Type Hinting.

Use British English in comments/plots ("Colour", "Normalised").

Document the "Why" (e.g., "Vertical flip used because universe is isotropic").