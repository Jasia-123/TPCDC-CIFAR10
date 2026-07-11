import os
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class ContrastiveTransform:
    """
    Generate two independently augmented views of one CIFAR-10 image.

    This is used during the representation-pretraining stage of the
    SCAN pipeline.
    """

    def __init__(self) -> None:
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size=32,
                scale=(0.2, 1.0),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.1,
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                CIFAR10_MEAN,
                CIFAR10_STD,
            ),
        ])

    def __call__(self, image):
        return (
            self.transform(image),
            self.transform(image),
        )


class SCANTransform:
    """
    Strong augmentation used while training SCAN's clustering stage.

    Feature extraction and final cluster prediction must continue
    using deterministic preprocessing instead.
    """

    def __init__(self) -> None:
        self.transform = transforms.Compose([
            transforms.RandomCrop(
                size=32,
                padding=4,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(
                num_ops=4,
                magnitude=10,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                CIFAR10_MEAN,
                CIFAR10_STD,
            ),
        ])

    def __call__(self, image):
        return self.transform(image)


class IndexedDataset(Dataset):
    """
    Wrap a dataset so that each item also returns its original index.

    The index is required when extracting features, mining nearest
    neighbours and later mapping SCAN assignments back to CIFAR-10.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample, target = self.dataset[index]

        return sample, target, index

    @property
    def classes(self):
        return getattr(self.dataset, "classes", None)

    @property
    def targets(self):
        return getattr(self.dataset, "targets", None)


def get_classifier_train_transform() -> Callable:
    """
    Augmentation used when training a supervised classifier.
    """
    return transforms.Compose([
        transforms.RandomCrop(
            size=32,
            padding=4,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            CIFAR10_MEAN,
            CIFAR10_STD,
        ),
    ])


def get_evaluation_transform() -> Callable:
    """
    Deterministic transform used for feature extraction and testing.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            CIFAR10_MEAN,
            CIFAR10_STD,
        ),
    ])


def get_cifar10_datasets(data_dir: str = "./data"):
    """
    Load datasets for supervised classifier training and evaluation.

    The training labels must not be accessed during TPCDC query
    selection. They are revealed only after image indices are selected.
    """

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_classifier_train_transform(),
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=get_evaluation_transform(),
    )

    return train_dataset, test_dataset


def get_pretraining_dataset(
    data_dir: str = "./data",
) -> Dataset:
    """
    Return CIFAR-10 with two-view contrastive augmentation.

    This dataset is used for the representation-pretraining stage
    preceding SCAN clustering.
    """

    return datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=ContrastiveTransform(),
    )


def get_scan_dataset(
    data_dir: str = "./data",
) -> Dataset:
    """
    Return indexed CIFAR-10 images with strong augmentation.

    This dataset is used only while training SCAN's clustering stage.
    CIFAR-10 labels are returned internally by torchvision but are
    ignored by SCAN.
    """

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=SCANTransform(),
    )

    return IndexedDataset(dataset)


def get_feature_dataset(
    data_dir: str = "./data",
    include_indices: bool = True,
) -> Dataset:
    """
    Return CIFAR-10 with deterministic preprocessing.

    This dataset is used for:
      - feature extraction;
      - nearest-neighbour mining;
      - SCAN cluster prediction;
      - typicality calculations.
    """

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_evaluation_transform(),
    )

    if include_indices:
        return IndexedDataset(dataset)

    return dataset


def create_data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
    drop_last: bool = False,
) -> DataLoader:
    """
    Construct a consistent PyTorch DataLoader.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def get_test_loader(
    test_dataset: Dataset,
    batch_size: int = 128,
    num_workers: int = 2,
) -> DataLoader:
    """
    Return the deterministic CIFAR-10 test loader.
    """

    return create_data_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )


if __name__ == "__main__":
    data_dir = os.environ.get(
        "CIFAR10_DATA_DIR",
        "./data",
    )

    train_dataset, test_dataset = get_cifar10_datasets(
        data_dir=data_dir,
    )

    pretraining_dataset = get_pretraining_dataset(
        data_dir=data_dir,
    )

    print(f"SCAN samples: {len(scan_dataset)}")

    scan_image, _, scan_index = scan_dataset[0]

    print(f"SCAN image shape: {scan_image.shape}")
    print(f"SCAN index: {scan_index}")

    feature_dataset = get_feature_dataset(
        data_dir=data_dir,
        include_indices=True,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Pretraining samples: {len(pretraining_dataset)}")
    print(f"Feature samples: {len(feature_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    train_image, train_label = train_dataset[0]

    (view_one, view_two), _ = pretraining_dataset[0]

    feature_image, feature_label, feature_index = (
        feature_dataset[0]
    )

    print(f"Classifier image shape: {train_image.shape}")
    print(f"Contrastive view 1 shape: {view_one.shape}")
    print(f"Contrastive view 2 shape: {view_two.shape}")
    print(f"Feature image shape: {feature_image.shape}")
    print(f"Feature index: {feature_index}")
    print(f"Example label: {feature_label}")
