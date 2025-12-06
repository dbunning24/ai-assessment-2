import torch
import torch.nn as nn
import torchvision.models as models


class SimCLR(nn.Module):
    """
    Pure PyTorch SimCLR model:
    - ResNet34 encoder (fc removed)
    - 2-layer projection head
    """

    def __init__(
        self,
        projection_dim: int = 128,
    ):
        super().__init__()

        # -------------------------
        # Encoder: ResNet-34
        # -------------------------
        resnet = models.resnet34(weights=None)

        # Make it friendlier for small-ish images (96x96)
        resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        resnet.maxpool = nn.Identity()

        # Remove classifier head
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        # ResNet34 final feature dim
        self.encoder_dim = 512

        # -------------------------
        # Projection Head (MLP)
        # -------------------------
        self.projector = nn.Sequential(
            nn.Linear(self.encoder_dim, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns encoder features (before projection)."""
        h = self.encoder(x)              # shape: (N, 512, 1, 1)
        h = h.squeeze(-1).squeeze(-1)    # shape: (N, 512)
        return h

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """Projects encoder features into contrastive embedding."""
        return self.projector(h)

    def forward(self, x: torch.Tensor):
        """
        Forward pass returns BOTH:
        - encoder features (h)
        - projected features (z)
        """
        h = self.encode(x)
        z = self.project(h)
        return h, z
