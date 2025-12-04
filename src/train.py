"""Training script for SimCLR on Galaxy Zoo 2 images.

This script orchestrates the full training pipeline, including data loading,
model initialisation, and trainer setup. It supports two hardware profiles:
Profile A (Desktop/GPU) for scientific runs and Profile B (Laptop/CPU)
for development.
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from PIL import Image

from dataset import GalaxyZooDataset
from model import SimCLR


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_normalization_stats(
    csv_path: str,
    image_dir: str,
    mapping_csv: str,
    n_samples: int = 10000,
) -> Tuple[list, list]:
    """Calculate mean and standard deviation from a subset of images.
    
    Args:
        csv_path: Path to gz2spec.csv.
        image_dir: Path to images directory.
        mapping_csv: Path to gz2maps.csv.
        n_samples: Number of images to sample for statistics.
        
    Returns:
        Tuple of (means, stds) for each channel.
    """
    logger.info(f'Computing normalisation statistics from {n_samples} images...')
    
    # Load data
    logger.info(f'  Loading morphology data from {csv_path}...')
    morph_data = pd.read_csv(csv_path)
    logger.info(f'  ✓ Loaded {len(morph_data)} galaxy records.')
    
    logger.info(f'  Scanning for present images in {image_dir}...')
    image_dir_path = Path(image_dir)
    present_ids = {
        int(f.stem) for f in image_dir_path.glob("*.jpg")
    }
    logger.info(f'  ✓ Found {len(present_ids)} image files.')
    
    logger.info(f'  Loading image mappings from {mapping_csv}...')
    mapping_data = pd.read_csv(mapping_csv)
    mapping_data = mapping_data[mapping_data['asset_id'].isin(present_ids)]
    logger.info(f'  ✓ Loaded {len(mapping_data)} mappings with available images.')
    
    logger.info('  Merging spectroscopic data with image mappings...')
    galaxy_data = morph_data.merge(
        mapping_data,
        left_on='dr7objid',
        right_on='objid',
        how='inner'
    )
    logger.info(f'  ✓ Merged to {len(galaxy_data)} galaxies.')
    
    if len(galaxy_data) > n_samples:
        logger.info(f'  Sampling {n_samples} galaxies...')
        galaxy_data = galaxy_data.sample(n=n_samples, random_state=42)
        logger.info(f'  ✓ Sampled to {len(galaxy_data)} galaxies.')
    
    galaxy_ids = galaxy_data['asset_id'].values
    
    # Accumulate statistics
    logger.info(f'  Computing statistics from {len(galaxy_ids)} images...')
    means = np.zeros(3)
    stds = np.zeros(3)
    
    for idx, galaxy_id in enumerate(galaxy_ids):
        if idx % max(1, len(galaxy_ids) // 5) == 0:
            logger.info(f'    Progress: [{idx}/{len(galaxy_ids)}]')
        image_path = image_dir_path / f"{galaxy_id}.jpg"
        if not image_path.exists():
            continue
        
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image) / 255.0
        
        means += image_array.mean(axis=(0, 1))
        stds += image_array.std(axis=(0, 1))
    
    means = (means / len(galaxy_ids)).tolist()
    stds = (stds / len(galaxy_ids)).tolist()
    
    logger.info(f'✓ Normalisation statistics computed:')
    logger.info(f'  Means (RGB): {[round(m, 4) for m in means]}')
    logger.info(f'  Stds (RGB): {[round(s, 4) for s in stds]}')
    
    return means, stds


def setup_profile(profile: str) -> dict:
    """Setup hardware profile configuration.
    
    Args:
        profile: Either 'A' (Desktop/GPU) or 'B' (Laptop/CPU).
        
    Returns:
        Dictionary with profile-specific settings.
    """
    if profile == 'A':
        logger.info('Profile A (Desktop/GPU) selected: Scientific run')
        return {
            'accelerator': 'gpu',
            'precision': '16-mixed',
            'batch_size': 256,
            'num_workers': 4,
            'max_epochs': 100,
            'n_samples': 100000,
        }
    elif profile == 'B':
        logger.info('Profile B (Laptop/CPU) selected: Development run (~5 mins)')
        return {
            'accelerator': 'auto',  # Auto-detect CPU/GPU
            'precision': '32',
            'batch_size': 32,
            'num_workers': 0,
            'max_epochs': 2,
            'n_samples': 1000,
        }
    else:
        raise ValueError(f"Unknown profile: {profile}. Choose 'A' or 'B'.")


def main() -> None:
    """Main training loop."""
    parser = argparse.ArgumentParser(
        description='Train SimCLR on Galaxy Zoo 2 images.'
    )
    parser.add_argument(
        '--profile',
        type=str,
        default='B',
        choices=['A', 'B'],
        help='Hardware profile: A (GPU/Scientific) or B (CPU/Development).',
    )
    parser.add_argument(
        '--fast-dev-run',
        action='store_true',
        help='Run on 1 batch to validate pipeline.',
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='../data/',
        help='Path to data directory.',
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=None,
        help='Override max epochs.',
    )
    
    args = parser.parse_args()
    
    # Setup profile
    profile_config = setup_profile(args.profile)
    
    # Override max_epochs if provided
    if args.max_epochs is not None:
        profile_config['max_epochs'] = args.max_epochs
    
    # Data paths
    data_path = Path(args.data_path)
    csv_path = str(data_path / 'gz2spec.csv')
    image_dir = str(data_path / 'images/')
    mapping_csv = str(data_path / 'gz2maps.csv')
    
    logger.info(f'Loading data from {data_path}')
    
    # Calculate normalisation statistics
    logger.info('Step 1/4: Computing normalisation statistics...')
    norm_stats = calculate_normalization_stats(
        csv_path, image_dir, mapping_csv, n_samples=10000
    )
    
    # Create dataset
    logger.info('Step 2/4: Creating dataset...')
    dataset = GalaxyZooDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        mapping_csv=mapping_csv,
        n_samples=profile_config['n_samples'],
        image_size=64,
        normalization_stats=norm_stats,
    )
    logger.info(f'✓ Dataset created with {len(dataset)} galaxies.')
    
    # Create data loader
    logger.info('Step 3/4: Creating data loader...')
    dataloader = DataLoader(
        dataset,
        batch_size=profile_config['batch_size'],
        shuffle=True,
        num_workers=profile_config['num_workers'],
        pin_memory=True,
    )
    logger.info(
        f"✓ Data loader created with batch_size={profile_config['batch_size']}"
    )
    
    # Create model
    logger.info('Step 4/4: Creating SimCLR model...')
    model = SimCLR(
        batch_size=profile_config['batch_size'],
        base_lr=1e-3,
        hidden_dim=512,
        projection_dim=128,
        temperature=0.5,
        image_size=64,
    )
    logger.info('✓ SimCLR model initialized.')
    
    # Create trainer
    logger.info('Creating PyTorch Lightning trainer...')
    logger.info(f'  Accelerator: {profile_config["accelerator"]}')
    logger.info(f'  Precision: {profile_config["precision"]}')
    logger.info(f'  Max epochs: {profile_config["max_epochs"]}')
    trainer = pl.Trainer(
        accelerator=profile_config['accelerator'],
        precision=profile_config['precision'],
        max_epochs=profile_config['max_epochs'],
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=10,
        enable_model_summary=True,
    )
    logger.info('✓ Trainer initialized.')
    
    # Train
    logger.info('='*60)
    logger.info('Starting training...')
    logger.info('='*60)
    trainer.fit(model, dataloader)
    
    logger.info('='*60)
    logger.info('✓ Training complete.')
    logger.info('='*60)


if __name__ == '__main__':
    main()
