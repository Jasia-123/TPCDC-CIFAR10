import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_datasets(data_dir="./data"):
    """
    Load CIFAR-10 training and test datasets.

    The training set represents the active-learning pool.
    Training labels must not be used during TPCDC query selection.
    """

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    return train_dataset, test_dataset


def get_test_loader(
    test_dataset,
    batch_size=128,
    num_workers=2,
):
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    import os

    data_dir = os.environ.get("CIFAR10_DATA_DIR", "./data")

    train_dataset, test_dataset = get_cifar10_datasets(
        data_dir=data_dir
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    image, label = train_dataset[0]

    print(f"Image shape: {image.shape}")
    print(f"Example label: {label}")
