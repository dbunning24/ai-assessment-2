import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Minimal NT-Xent loss for SimCLR.
    Computes the contrastive loss between two batches of projected features.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        # Normalise projections to unit vectors
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        batch_size = z_i.size(0)

        # Concatenate: (2N, D)
        z = torch.cat([z_i, z_j], dim=0)

        # Cosine similarity matrix: (2N, 2N)
        similarity_matrix = torch.mm(z, z.t()) / self.temperature

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

        # Positive pairs: i ↔ i + batch_size
        positives = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device)
        ])

        # Labels = index of the positive example for each row
        labels = positives

        # Cross-entropy over similarity rows
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss
