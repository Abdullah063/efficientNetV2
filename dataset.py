import os
from torch.utils.data import DataLoader
from torchvision import datasets
from config import *


def get_dataloaders():
    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, 'train'),
        transform=train_transforms
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, 'val'),
        transform=val_transforms
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, 'test'),
        transform=val_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_dataset.classes