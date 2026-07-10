import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models import SimCLRModel
import numpy as np


def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)

    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)

    similarity = torch.matmul(z, z.T) / temperature

    mask = torch.eye(
        2 * batch_size,
        dtype=torch.bool,
        device=z.device,
    )

    similarity = similarity.masked_fill(mask, float("-inf"))

    positive_indices = torch.arange(
        batch_size,
        device=z.device,
    )

    targets = torch.cat([
        positive_indices + batch_size,
        positive_indices,
    ])

    return F.cross_entropy(similarity, targets)


def train_simclr(
    dataset,
    device,
    epochs=50,
    batch_size=512,
    learning_rate=0.4,
    momentum=0.9,
    weight_decay=1e-4,
    temperature=0.5,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    model = SimCLRModel().to(device)

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

        for (view1, view2), _ in loader:
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)

            optimizer.zero_grad()

            _, z1 = model(view1)
            _, z2 = model(view2)

            loss = nt_xent_loss(
                z1,
                z2,
                temperature=temperature,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        average_loss = total_loss / len(loader)

        print(
            f"SimCLR Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {average_loss:.4f}"
        )

    return model


@torch.no_grad()
def extract_embeddings(
    model,
    dataset,
    device,
    batch_size=512,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model.eval()

    all_embeddings = []

    for images, _ in loader:
        images = images.to(device, non_blocking=True)

        features = model.encoder(images)

        features = F.normalize(
            features,
            p=2,
            dim=1,
        )

        all_embeddings.append(
            features.cpu().numpy()
        )

    embeddings = np.concatenate(
        all_embeddings,
        axis=0,
    )

    return embeddings
