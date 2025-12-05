import os
from pathlib import Path
from typing import Tuple

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class GalaxyZooDataset(Dataset):
    """
    Minimal, clean dataset class for SimCLR training.
    Loads galaxy images, applies preprocessing, and generates
    two augmented views of each image.
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        mapping_csv: str,
        n_samples: int = 100000,
        image_size: int = 64,
    ):
        self.image_dir = Path(image_dir)
        self.image_size = image_size

        # -------------------------------
        # Load metadata and image mapping
        # -------------------------------
        morph = pd.read_csv(csv_path)
        mapping = pd.read_csv(mapping_csv)

        # Only keep image IDs that actually exist
        available_ids = {int(f.stem) for f in self.image_dir.glob("*.jpg")}
        mapping = mapping[mapping["asset_id"].isin(available_ids)]

        # Merge metadata with available image IDs
        merged = morph.merge(
            mapping,
            left_on="dr7objid",
            right_on="objid",
            how="inner"
        )

        # Subsample for feasibility
        if len(merged) > n_samples:
            merged = merged.sample(n=n_samples, random_state=42)

        self.galaxy_ids = merged["asset_id"].values

        # -------------------------------
        # Preprocessing (centre crop + resize)
        # -------------------------------
        self.base_transform = T.Compose([
            T.CenterCrop(160),       # images are ~256px; this reduces noise
            T.Resize(image_size),
            T.ToTensor(),
        ])

        # -------------------------------
        # Augmentations for SimCLR
        # -------------------------------
        self.augment = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
        ])

    def __len__(self):
        return len(self.galaxy_ids)

    def load_image(self, galaxy_id: int) -> Image.Image:
        img_path = self.image_dir / f"{galaxy_id}.jpg"
        img = Image.open(img_path).convert("RGB")
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        galaxy_id = self.galaxy_ids[idx]

        # Load + preprocessing
        img = self.load_image(galaxy_id)
        img = self.base_transform(img)

        # Two augmented views
        view1 = self.augment(img)
        view2 = self.augment(img)

        return view1, view2, int(galaxy_id)
