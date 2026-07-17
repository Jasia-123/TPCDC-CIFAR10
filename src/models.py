from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ResNet18Encoder(nn.Module):
    """
    ResNet-18 feature encoder adapted for CIFAR-10.

    The standard ImageNet 7x7 convolution and max-pooling layer
    are replaced because CIFAR-10 images are only 32x32 pixels.
    """

    def __init__(self) -> None:
        super().__init__()

        backbone = resnet18(weights=None)

        backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        backbone.maxpool = nn.Identity()

        # Remove the original ImageNet classification layer.
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.feature_dim = 512

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Return the 512-dimensional penultimate representation.
        """
        return self.backbone(images)


class CIFAR10Classifier(nn.Module):
    """
    ResNet-18 classifier used to evaluate selected labelled sets.

    A fresh instance must be created for every active-learning
    round so that evaluation does not carry information between
    cumulative budgets.
    """

    def __init__(
        self,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError(
                "num_classes must be positive."
            )

        self.encoder = ResNet18Encoder()

        self.classifier = nn.Linear(
            self.encoder.feature_dim,
            num_classes,
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        features = self.encoder(images)

        return self.classifier(features)


class ContrastivePretrainModel(nn.Module):
    """
    Contrastive representation model used as SCAN's pretext stage.

    This is not the complete TPCRP algorithm. For TPCDC, the trained
    encoder is subsequently passed to the learnable SCAN clustering
    stage rather than K-means.
    """

    def __init__(self, projection_dim: int = 128) -> None:
        super().__init__()

        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive.")

        self.encoder = ResNet18Encoder()

        self.projection_head = nn.Sequential(
            nn.Linear(
                self.encoder.feature_dim,
                self.encoder.feature_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                self.encoder.feature_dim,
                projection_dim,
            ),
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return raw encoder features and normalized projections.

        Returns
        -------
        features:
            Tensor with shape [batch_size, 512].

        projections:
            L2-normalized tensor with shape
            [batch_size, projection_dim].
        """
        features = self.encoder(images)

        projections = self.projection_head(features)
        projections = F.normalize(
            projections,
            p=2,
            dim=1,
        )

        return features, projections

    @torch.no_grad()
    def encode(
        self,
        images: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Extract encoder representations without using the projection head.
        """
        features = self.encoder(images)

        if normalize:
            features = F.normalize(
                features,
                p=2,
                dim=1,
            )

        return features


class SCANModel(nn.Module):
    """
    Learnable SCAN clustering model.

    The model starts from a contrastively pretrained ResNet-18
    encoder and adds one or more trainable cluster-classification
    heads. TPCDC uses the resulting SCAN predictions as its cluster
    assignments.

    The backbone remains trainable during SCAN clustering.
    """

    def __init__(
        self,
        encoder: ResNet18Encoder,
        num_clusters: int,
        num_heads: int = 1,
    ) -> None:
        super().__init__()

        if num_clusters <= 1:
            raise ValueError(
                "num_clusters must be greater than one."
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be positive."
            )

        self.encoder = encoder
        self.num_clusters = num_clusters
        self.num_heads = num_heads

        self.cluster_heads = nn.ModuleList([
            nn.Linear(
                self.encoder.feature_dim,
                num_clusters,
            )
            for _ in range(num_heads)
        ])

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False,
    ):
        """
        Produce cluster logits for each SCAN head.

        Returns a list because SCAN can support multiple heads.
        Our TPCDC implementation will initially use one head.
        """
        features = self.encoder(images)

        outputs = [
            head(features)
            for head in self.cluster_heads
        ]

        if return_features:
            return outputs, features

        return outputs

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        head_index: int = 0,
    ) -> torch.Tensor:
        """
        Return the predicted cluster index for every image.
        """
        if not 0 <= head_index < self.num_heads:
            raise IndexError(
                f"head_index must be between 0 and "
                f"{self.num_heads - 1}."
            )

        self.eval()

        logits = self.forward(images)[head_index]

        return torch.argmax(
            logits,
            dim=1,
        )


def build_scan_model(
    pretrained_model: ContrastivePretrainModel,
    num_clusters: int,
    num_heads: int = 1,
) -> SCANModel:
    """
    Create a fresh SCAN model from a pretrained representation model.

    A deep copy is used so that training SCAN does not overwrite the
    reusable contrastive-pretraining checkpoint. This is important
    because TPCDC creates a new cluster count at each active-learning
    round.
    """
    if not isinstance(
        pretrained_model,
        ContrastivePretrainModel,
    ):
        raise TypeError(
            "pretrained_model must be a "
            "ContrastivePretrainModel."
        )

    encoder_copy = deepcopy(
        pretrained_model.encoder
    )

    return SCANModel(
        encoder=encoder_copy,
        num_clusters=num_clusters,
        num_heads=num_heads,
    )
