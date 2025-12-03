"""NT-Xent (Normalised Temperature-scaled Cross Entropy) Loss for SimCLR.

This module implements the symmetric NT-Xent loss used in SimCLR.
The loss maximises agreement between positive pairs (the same image with
different augmentations) whilst minimising agreement with negative pairs
(different images).
"""

from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn


class NTXentLoss(nn.Module):
    """Normalised Temperature-scaled Cross Entropy Loss.
    
    Implements the contrastive loss used in SimCLR. For each image in a batch,
    its augmented view is treated as a positive pair, and all other images
    are treated as negative pairs.
    
    The loss is computed as:
        loss = -log( exp(sim(z_i, z_j) / tau) / sum_k exp(sim(z_i, z_k) / tau) )
    
    Where sim is cosine similarity and tau is the temperature parameter.
    """
    
    def __init__(self, temperature: float = 0.5) -> None:
        """Initialise the loss function.
        
        Args:
            temperature: Temperature parameter tau. Lower values sharpen
                        the distribution (default 0.5).
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NT-Xent loss for a batch of projections.
        
        Args:
            z_i: Projection head output for first view, shape (batch_size, dim).
            z_j: Projection head output for second view, shape (batch_size, dim).
            
        Returns:
            Scalar loss value.
        """
        # Normalise projections
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        batch_size = z_i.shape[0]
        
        # Concatenate z_i and z_j: shape (2*batch_size, dim)
        z = torch.cat([z_i, z_j], dim=0)
        
        # Compute cosine similarity matrix: (2*batch_size, 2*batch_size)
        similarity_matrix = torch.mm(z, z.T)
        
        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create labels: positive pairs are at (i, i+batch_size) and vice versa
        # Labels are the indices of positive pairs
        labels = torch.arange(batch_size, dtype=torch.long, device=z.device)
        labels = torch.cat([
            labels + batch_size,  # For z_i views, positives are in z_j
            labels  # For z_j views, positives are in z_i
        ])
        
        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
