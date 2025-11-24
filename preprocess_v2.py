"""Data loading utilities used by training and inference.

Provides the `get_loaders` function which returns PyTorch DataLoaders for
train/val/test splits and the class ordering used by ImageFolder.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_loaders(data_dir: str = "dataset/DATASET", batch_size: int = 32, val_ratio: float = 0.15):
    """Return (train_loader, val_loader, test_loader, classes).

    Parameters
    - data_dir: Path to the dataset root that contains `train/` and `test/` subfolders.
    - batch_size: Batch size for loaders.
    - val_ratio: Fraction of the training set to hold out as validation.
    """

    # Transforms: match the spec provided (3-channel normalization)
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Root is .../train and .../test, not the class folder
    full_train = datasets.ImageFolder(
        root=f"{data_dir}/train", transform=train_transform)
    test_data = datasets.ImageFolder(
        root=f"{data_dir}/test", transform=test_transform)

    # Split into train / val
    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size
    train_data, val_data = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    classes = full_train.classes  # e.g. ['FAKE', 'REAL']
    return train_loader, val_loader, test_loader, classes


if __name__ == "__main__":
    tl, vl, test, classes = get_loaders()
    print("Classes detected:", classes)
    print("Train samples:", len(tl.dataset))
    print("Val samples:", len(vl.dataset))
    print("Test samples:", len(test.dataset))
