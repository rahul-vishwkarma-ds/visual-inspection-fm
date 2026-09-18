"""Frozen vision-backbone feature extraction.

Pulled out of the original sbi_prototype.py so the backbone can be reused by
both the SBI model (modeling/sbi_model.py) and, later, simulate.py's
image -> embedding pipeline for synthetic defect samples.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from visual_inspection_fm.config import IMAGE_SIZE

EMBEDDING_DIM = 512  # ResNet18's penultimate-layer width


def load_frozen_backbone() -> nn.Sequential:
    """Load a frozen, pretrained ResNet18 with the classification head removed.

    Stands in for "the Bosch foundation model backbone" for hackathon purposes —
    swap this for the real pre-trained backbone once/if one becomes available.
    """
    print("Loading frozen vision backbone...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone = nn.Sequential(*list(resnet.children())[:-1])

    for param in backbone.parameters():
        param.requires_grad = False

    return backbone


@torch.no_grad()
def embed(backbone: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
    """Run a batch of images [B, 3, IMAGE_SIZE, IMAGE_SIZE] through the frozen
    backbone and return flattened embeddings [B, EMBEDDING_DIM]."""
    embeddings = backbone(x)
    return embeddings.view(embeddings.size(0), -1)


def mock_camera_image(batch_size: int = 1) -> torch.Tensor:
    """A stand-in for a real image captured from the edge appliance's camera,
    until simulate.py's synthetic defect generator is wired in."""
    return torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
