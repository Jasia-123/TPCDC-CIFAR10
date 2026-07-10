import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class SimCLRTransform:
    def __init__(self):
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


def get_simclr_dataset(data_dir="./data"):
    return datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=SimCLRTransform(),
    )


def get_embedding_dataset(data_dir="./data"):
    embedding_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            CIFAR10_MEAN,
            CIFAR10_STD,
        ),
    ])

    return datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=embedding_transform,
    )


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
