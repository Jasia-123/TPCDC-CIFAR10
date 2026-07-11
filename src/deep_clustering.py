from copy import deepcopy
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data import create_data_loader
from src.models import (
    ContrastivePretrainModel,
    SCANModel,
)


# Contrastive representation pretraining


def nt_xent_loss(
    first_projections: torch.Tensor,
    second_projections: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """
    Compute the normalized temperature-scaled cross-entropy loss.

    Each image has two augmented views. The matching view is treated
    as the positive example; all other views in the batch are negatives.
    """

    if first_projections.shape != second_projections.shape:
        raise ValueError(
            "The two projection tensors must have identical shapes."
        )

    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    batch_size = first_projections.size(0)

    projections = torch.cat(
        [first_projections, second_projections],
        dim=0,
    )
    projections = F.normalize(projections, p=2, dim=1)

    similarity = torch.matmul(
        projections,
        projections.T,
    )
    similarity = similarity / temperature

    # Prevent each representation from matching itself.
    diagonal_mask = torch.eye(
        2 * batch_size,
        dtype=torch.bool,
        device=similarity.device,
    )
    similarity = similarity.masked_fill(
        diagonal_mask,
        float("-inf"),
    )

    positive_targets = torch.arange(
        2 * batch_size,
        device=similarity.device,
    )
    positive_targets = (
        positive_targets + batch_size
    ) % (2 * batch_size)

    return F.cross_entropy(
        similarity,
        positive_targets,
    )


def train_contrastive_pretraining(
    dataset: Dataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    temperature: float,
    projection_dim: int,
    num_workers: int = 2,
) -> ContrastivePretrainModel:
    """
    Train the self-supervised representation stage used before SCAN.

    The resulting encoder is subsequently used for neighbour mining
    and SCAN clustering. No K-means is used.
    """

    loader = create_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    model = ContrastivePretrainModel(
        projection_dim=projection_dim,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_batches = 0

        for (first_view, second_view), _ in loader:
            first_view = first_view.to(
                device,
                non_blocking=True,
            )
            second_view = second_view.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            _, first_projections = model(first_view)
            _, second_projections = model(second_view)

            loss = nt_xent_loss(
                first_projections=first_projections,
                second_projections=second_projections,
                temperature=temperature,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        scheduler.step()

        average_loss = total_loss / max(total_batches, 1)

        print(
            f"Pretraining epoch [{epoch + 1}/{epochs}] "
            f"loss: {average_loss:.4f}"
        )

    return model


# Feature extraction


@torch.no_grad()
def extract_features(
    model: ContrastivePretrainModel,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 512,
    num_workers: int = 2,
) -> np.ndarray:
    """
    Extract L2-normalized 512-dimensional encoder features.

    The supplied dataset must return:
        image, target, original_index

    Features are returned in original CIFAR-10 index order.
    """

    loader = create_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    model.eval()

    feature_dimension = model.encoder.feature_dim
    all_features = np.empty(
        (len(dataset), feature_dimension),
        dtype=np.float32,
    )

    for images, _, indices in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        features = model.encode(
            images,
            normalize=True,
        )

        indices = indices.cpu().numpy()
        all_features[indices] = features.cpu().numpy()

    return all_features


# Nearest-neighbour mining


def mine_nearest_neighbours(
    features: np.ndarray,
    num_neighbours: int = 20,
) -> np.ndarray:
    """
    Mine nearest neighbours in the normalized representation space.

    The image itself is removed from its neighbour list.

    Returns
    -------
    neighbour_indices:
        Integer array with shape
        [number_of_samples, num_neighbours].
    """

    if features.ndim != 2:
        raise ValueError(
            "features must have shape [samples, dimensions]."
        )

    if num_neighbours <= 0:
        raise ValueError(
            "num_neighbours must be positive."
        )

    num_samples = features.shape[0]

    if num_neighbours >= num_samples:
        raise ValueError(
            "num_neighbours must be smaller than the dataset."
        )

    # Since features are L2-normalized, cosine distance is suitable.
    nearest_neighbours = NearestNeighbors(
        n_neighbors=num_neighbours + 1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    nearest_neighbours.fit(features)

    _, indices = nearest_neighbours.kneighbors(
        features,
        return_distance=True,
    )

    # Column zero is normally the sample itself.
    neighbour_indices = indices[:, 1:]

    if neighbour_indices.shape != (
        num_samples,
        num_neighbours,
    ):
        raise RuntimeError(
            "Unexpected nearest-neighbour array shape."
        )

    return neighbour_indices.astype(np.int64)


# Dataset used for SCAN neighbour-consistency training


class NeighbourPairDataset(Dataset):
    """
    Return an image and one randomly selected mined neighbour.

    `dataset` should return:
        transformed_image, target, original_index

    Labels are ignored and are never used by SCAN.
    """

    def __init__(
        self,
        dataset: Dataset,
        neighbour_indices: np.ndarray,
    ) -> None:
        if len(dataset) != neighbour_indices.shape[0]:
            raise ValueError(
                "Dataset length and neighbour array must match."
            )

        if neighbour_indices.ndim != 2:
            raise ValueError(
                "neighbour_indices must be a two-dimensional array."
            )

        self.dataset = dataset
        self.neighbour_indices = neighbour_indices

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        anchor_image, _, anchor_index = self.dataset[index]

        neighbour_position = random.randrange(
            self.neighbour_indices.shape[1]
        )

        neighbour_index = int(
            self.neighbour_indices[
                anchor_index,
                neighbour_position,
            ]
        )

        neighbour_image, _, returned_index = (
            self.dataset[neighbour_index]
        )

        if returned_index != neighbour_index:
            raise RuntimeError(
                "Dataset indices are not aligned with CIFAR-10 indices."
            )

        return (
            anchor_image,
            neighbour_image,
            anchor_index,
            neighbour_index,
        )


# SCAN objective


def entropy_of_mean_distribution(
    probabilities: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Calculate entropy of the mean cluster distribution.

    Maximizing this entropy discourages collapse into one cluster.
    """

    mean_probability = probabilities.mean(dim=0)

    return -torch.sum(
        mean_probability
        * torch.log(mean_probability + epsilon)
    )


def scan_loss(
    anchor_logits: torch.Tensor,
    neighbour_logits: torch.Tensor,
    entropy_weight: float,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SCAN's neighbour-consistency and entropy objective.

    Neighbour consistency encourages an image and its mined neighbour
    to have the same cluster distribution. Entropy regularization
    discourages assigning every image to one cluster.
    """

    if anchor_logits.shape != neighbour_logits.shape:
        raise ValueError(
            "Anchor and neighbour logits must have identical shapes."
        )

    anchor_probabilities = F.softmax(
        anchor_logits,
        dim=1,
    )
    neighbour_probabilities = F.softmax(
        neighbour_logits,
        dim=1,
    )

    similarity = torch.sum(
        anchor_probabilities * neighbour_probabilities,
        dim=1,
    )

    similarity = torch.clamp(
        similarity,
        min=epsilon,
        max=1.0,
    )

    consistency_loss = -torch.log(similarity).mean()

    entropy = entropy_of_mean_distribution(
        anchor_probabilities,
        epsilon=epsilon,
    )

    total_loss = (
        consistency_loss
        - entropy_weight * entropy
    )

    return total_loss, consistency_loss, entropy


# SCAN training


def train_scan(
    model: SCANModel,
    dataset: Dataset,
    neighbour_indices: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    entropy_weight: float,
    num_workers: int = 2,
) -> SCANModel:
    """
    Train SCAN's first clustering stage.

    The self-labelling/pseudo-labelling stage is intentionally omitted,
    matching the TPCDC implementation described in the paper.
    """

    pair_dataset = NeighbourPairDataset(
        dataset=dataset,
        neighbour_indices=neighbour_indices,
    )

    loader = create_data_loader(
        dataset=pair_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_loss = float("inf")
    best_state_dict = None
    best_epoch = None

    for epoch in range(epochs):
        model.train()

        total_loss_value = 0.0
        total_consistency = 0.0
        total_entropy = 0.0
        total_batches = 0

        for (
            anchor_images,
            neighbour_images,
            _,
            _,
        ) in loader:
            anchor_images = anchor_images.to(
                device,
                non_blocking=True,
            )
            neighbour_images = neighbour_images.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            anchor_outputs = model(anchor_images)
            neighbour_outputs = model(neighbour_images)

            head_losses = []
            head_consistency_losses = []
            head_entropies = []

            for anchor_logits, neighbour_logits in zip(
                anchor_outputs,
                neighbour_outputs,
            ):
                (
                    head_loss,
                    consistency_loss,
                    entropy,
                ) = scan_loss(
                    anchor_logits=anchor_logits,
                    neighbour_logits=neighbour_logits,
                    entropy_weight=entropy_weight,
                )

                head_losses.append(head_loss)
                head_consistency_losses.append(
                    consistency_loss
                )
                head_entropies.append(entropy)

            loss = torch.stack(head_losses).mean()

            loss.backward()
            optimizer.step()

            total_loss_value += loss.item()
            total_consistency += torch.stack(
                head_consistency_losses
            ).mean().item()
            total_entropy += torch.stack(
                head_entropies
            ).mean().item()
            total_batches += 1

        denominator = max(total_batches, 1)

        average_loss = total_loss_value / denominator
        average_consistency = total_consistency / denominator
        average_entropy = total_entropy / denominator

        if not np.isfinite(average_loss):
            raise RuntimeError(
                f"Non-finite SCAN loss encountered at epoch {epoch + 1}."
            )

        if average_loss < best_loss:
            best_loss = average_loss
            best_epoch = epoch + 1
            best_state_dict = deepcopy(model.state_dict())

        print(
            f"SCAN epoch [{epoch + 1}/{epochs}] "
            f"loss: {average_loss:.4f} | "
            f"consistency: {average_consistency:.4f} | "
            f"entropy: {average_entropy:.4f} | "
            f"best epoch: {best_epoch}"
        )

    if best_state_dict is None:
        raise RuntimeError(
            "SCAN training did not produce a valid model state."
        )

    model.load_state_dict(best_state_dict)

    print(
        f"Restored lowest-loss SCAN model from epoch "
        f"{best_epoch} with loss {best_loss:.4f}."
    )

    return model


# Cluster prediction


@torch.no_grad()
def predict_cluster_assignments(
    model: SCANModel,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 512,
    num_workers: int = 2,
    head_index: int = 0,
) -> np.ndarray:
    """
    Predict a SCAN cluster assignment for every CIFAR-10 image.

    Assignments are returned in original dataset-index order.
    """

    if not 0 <= head_index < model.num_heads:
        raise IndexError(
            "head_index is outside the available SCAN heads."
        )

    loader = create_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    assignments = np.empty(
        len(dataset),
        dtype=np.int64,
    )

    model.eval()

    for images, _, indices in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        outputs = model(images)
        logits = outputs[head_index]

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        indices = indices.cpu().numpy()
        assignments[indices] = (
            predictions.cpu().numpy()
        )

    return assignments
