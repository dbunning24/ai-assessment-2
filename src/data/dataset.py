from pathlib import Path
from typing import Tuple

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class GalaxyZooDataset(Dataset):
    """
    Dataset class for SimCLR training on Galaxy Zoo.
    Loads galaxy images, applies preprocessing, and generates
    two augmented views of each image.
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        mapping_csv: str,
        n_samples: int = 15000,
        image_size: int = 96,
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
        # Full transform: crop + aug + tensor
        # -------------------------------
        self.transform = T.Compose([
            # work on central region to reduce junk
            T.CenterCrop(160),

            # main SimCLR crop at target resolution
            T.RandomResizedCrop(self.image_size, scale=(0.5, 1.0)),

            # geometry
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(10),

            # gentle photometry
            T.ColorJitter(
                brightness=0.05,
                contrast=0.05,
                saturation=0.05,
                hue=0.0,
            ),

            # mild blur for SimCLR invariance
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),

            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.galaxy_ids)

    def load_image(self, galaxy_id: int) -> Image.Image:
        img_path = self.image_dir / f"{galaxy_id}.jpg"
        img = Image.open(img_path).convert("RGB")
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        galaxy_id = self.galaxy_ids[idx]

        img = self.load_image(galaxy_id)

        # Two independently augmented views
        view1 = self.transform(img)
        view2 = self.transform(img)

        return view1, view2, int(galaxy_id)
