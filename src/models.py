import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = resnet18(weights=None)

        # CIFAR-10 adaptation: 32x32 images
        backbone.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        backbone.maxpool = nn.Identity()

        self.encoder = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.feature_dim = 512

    def forward(self, x):
        features = self.encoder(x)
        return torch.flatten(features, 1)


class SimCLRModel(nn.Module):
    def __init__(self, projection_dim=128):
        super().__init__()

        self.encoder = ResNet18Encoder()

        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim),
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projector(features)

        projections = F.normalize(
            projections,
            dim=1,
        )

        return features, projections
