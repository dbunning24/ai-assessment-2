"""Galaxy Zoo Dataset for SimCLR contrastive learning.

This module provides the GalaxyZooDataset class which loads galaxy images
and applies augmentations to generate two views for contrastive learning.
The universe is isotropic, so we apply horizontal and vertical flips.
Colour is physically meaningful, so we exclude grayscale conversion.
"""

from typing import Tuple
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class GalaxyZooDataset(Dataset):
    """Dataset for Galaxy Zoo 2 images with augmentations.
    
    Loads galaxy images and applies identical augmentation pipelines to create
    two views for SimCLR contrastive learning. Returns both views alongside
    the galaxy identifier for later validation.
    """
    
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        mapping_csv: str,
        n_samples: int = 100000,
        image_size: int = 64,
        normalization_stats: Tuple[list, list] | None = None,
    ) -> None:
        """Initialise the dataset.
        
        Args:
            csv_path: Path to gz2spec.csv containing galaxy morphology data.
            image_dir: Path to directory containing galaxy image files.
            mapping_csv: Path to gz2maps.csv mapping asset_id to images.
            n_samples: Number of galaxies to sample (default 100k).
            image_size: Target image size (default 64x64).
            normalization_stats: Tuple of (means, stds) for normalisation.
                                If None, will use ImageNet defaults.
        """
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        
        # Load and filter data
        morph_data = pd.read_csv(csv_path)
        morph_data = morph_data[morph_data['z'] > 0]  # Extragalactic only
        
        # Get available images
        present_ids = {
            int(f.stem) for f in self.image_dir.glob("*.jpg")
        }
        
        mapping_data = pd.read_csv(mapping_csv)
        mapping_data = mapping_data[mapping_data['asset_id'].isin(present_ids)]
        
        # Merge and sample
        self.galaxy_data = morph_data.merge(
            mapping_data,
            left_on='dr7objid',
            right_on='dr7_objid',
            how='inner'
        )
        
        if len(self.galaxy_data) > n_samples:
            self.galaxy_data = self.galaxy_data.sample(n=n_samples, random_state=42)
        
        self.galaxy_ids = self.galaxy_data['asset_id'].values
        
        # Normalisation statistics
        if normalization_stats is None:
            # ImageNet defaults
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            self.mean, self.std = normalization_stats
        
        # Define augmentation pipeline
        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(
                size=image_size,
                scale=(0.5, 1.0),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomHorizontalFlip(p=0.5),  # Universe is isotropic
            transforms.RandomVerticalFlip(p=0.5),    # Universe is isotropic
            transforms.RandomRotation(degrees=(90, 180, 270)),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.mean,
                std=self.std
            ),
        ])
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.galaxy_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Load and augment a galaxy image.
        
        Args:
            idx: Index of galaxy to load.
            
        Returns:
            Tuple of (view1, view2, galaxy_id) where view1 and view2 are
            differently augmented versions of the same image.
        """
        galaxy_id = self.galaxy_ids[idx]
        image_path = self.image_dir / f"{galaxy_id}.jpg"
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply augmentations to create two views
        view1 = self.augmentation(image)
        view2 = self.augmentation(image)
        
        return view1, view2, int(galaxy_id)
