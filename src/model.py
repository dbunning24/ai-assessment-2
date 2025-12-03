"""SimCLR model implemented as a PyTorch Lightning module.

This module implements the SimCLR contrastive learning framework using
PyTorch Lightning for streamlined training and logging.
"""

from typing import Any, Dict
import torch
import torch.nn as nn
from torch.optim import Adam
import pytorch_lightning as pl
import torchvision.models as models

from loss import NTXentLoss


class SimCLR(pl.LightningModule):
    """SimCLR model for unsupervised representation learning.
    
    Combines a ResNet18 encoder with a projection head. The model learns
    representations by maximising agreement between differently augmented
    views of the same image using contrastive learning.
    """
    
    def __init__(
        self,
        batch_size: int = 256,
        base_lr: float = 1e-3,
        hidden_dim: int = 512,
        projection_dim: int = 128,
        temperature: float = 0.5,
        image_size: int = 64,
    ) -> None:
        """Initialise the SimCLR module.
        
        Args:
            batch_size: Batch size for linear learning rate scaling.
            base_lr: Base learning rate (default 1e-3).
            hidden_dim: Hidden dimension of ResNet18 encoder (default 512).
            projection_dim: Projection head output dimension (default 128).
            temperature: Temperature parameter for NT-Xent loss (default 0.5).
            image_size: Input image size (default 64).
        """
        super().__init__()
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.image_size = image_size
        
        # Save hyperparameters for logging
        self.save_hyperparameters()
        
        # Encoder: ResNet18 without classification head
        self.encoder = models.resnet18(weights=None)
        
        # Adjust first conv layer if images are small (< 96px)
        if image_size < 96:
            self.encoder.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
        
        # Remove classification head, use features only
        self.encoder.fc = nn.Identity()
        
        # Projection head: MLP (hidden_dim -> hidden_dim -> projection_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )
        
        # Loss function
        self.criterion = NTXentLoss(temperature=temperature)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder (feature extraction).
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W).
            
        Returns:
            Feature representation from encoder, shape (batch_size, hidden_dim).
        """
        return self.encoder(x)
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step for a single batch.
        
        Args:
            batch: Tuple of (view1, view2, galaxy_id).
            batch_idx: Batch index.
            
        Returns:
            Scalar loss value.
        """
        view1, view2, _ = batch
        
        # Extract features
        h_i = self.encoder(view1)
        h_j = self.encoder(view2)
        
        # Project features
        z_i = self.projection_head(h_i)
        z_j = self.projection_head(h_j)
        
        # Compute contrastive loss
        loss = self.criterion(z_i, z_j)
        
        # Log to TensorBoard
        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        return loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer with linear learning rate scaling.
        
        Implements Linear Scaling Rule:
            LR = base_lr * (batch_size / 256)
        
        This maintains training stability across different batch sizes,
        as SimCLR performance is directly tied to batch size.
        
        Returns:
            Dictionary with optimizer configuration.
        """
        # Apply linear scaling rule
        scaled_lr = self.base_lr * (self.batch_size / 256.0)
        
        self.log('learning_rate', scaled_lr)
        
        optimizer = Adam(
            self.parameters(),
            lr=scaled_lr,
        )
        
        return {
            'optimizer': optimizer,
        }
