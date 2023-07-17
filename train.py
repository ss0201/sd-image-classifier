import argparse
import logging
import os
import sys
from typing import List, Tuple, cast

import torch
import torch.utils.data
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_V2_M_Weights


def train(
    data_dir: str, model_dir: str, epochs: int, batch_size: int, n_splits: int
) -> None:
    device = get_device()
    train_transform, val_transform = get_transforms()
    full_dataset = datasets.ImageFolder(data_dir)
    folds = get_datasets(full_dataset, n_splits, train_transform, val_transform)
    model = get_model(device, len(full_dataset.classes))
    criterion = get_criterion(device, full_dataset)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    for fold, (train_dataset, val_dataset) in enumerate(folds):
        logging.info(f"Starting fold {fold + 1}...")
        train_dataloader, val_dataloader = get_dataloaders(
            train_dataset, val_dataset, batch_size
        )
        for epoch in range(epochs):
            logging.info(f"Starting epoch {epoch + 1}...")
            train_epoch(device, model, criterion, optimizer, train_dataloader)
            val_loss = validate_epoch(device, model, criterion, val_dataloader)
            scheduler.step(val_loss)

        logging.info(f"Finished fold {fold + 1}.")

    save_model(model_dir, model, full_dataset)


def get_device() -> torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}.")
    return device


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


def get_datasets(
    full_dataset: datasets.ImageFolder,
    n_splits: int,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
) -> List[Tuple[Dataset, Dataset]]:
    kfold = KFold(n_splits=n_splits)
    indices = [str(i) for i in range(len(full_dataset))]
    folds = []
    for train_indices, val_indices in kfold.split(indices):
        train_indices = cast(List[int], train_indices)
        val_indices = cast(List[int], val_indices)
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)

        train_subset.dataset = cast(datasets.ImageFolder, train_subset.dataset)
        val_subset.dataset = cast(datasets.ImageFolder, val_subset.dataset)
        train_subset.dataset.transform = train_transform
        val_subset.dataset.transform = val_transform

        folds.append((train_subset, val_subset))

    return folds


def get_dataloaders(
    train_dataset: Dataset, val_dataset: Dataset, batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader


def get_model(device: torch.device, num_classes: int) -> nn.Module:
    model = models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, num_classes),
    )
    model = model.to(device)

    return model


def get_criterion(
    device: torch.device, full_dataset: datasets.ImageFolder
) -> nn.CrossEntropyLoss:
    class_count = [0] * len(full_dataset.classes)
    for _, class_idx in full_dataset:
        class_count[class_idx] += 1
    class_weights = 1.0 / torch.tensor(class_count, dtype=torch.float)
    class_weights_normalized = (class_weights / class_weights.sum()).to(device)

    return nn.CrossEntropyLoss(weight=class_weights_normalized)


def get_optimizer(model: nn.Module) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=0.001)


def get_scheduler(optimizer: optim.Optimizer) -> ReduceLROnPlateau:
    return ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)


def train_epoch(
    device: torch.device,
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Optimizer,
    train_dataloader: DataLoader,
) -> float:
    model.train()
    train_loss = 0.0
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss: torch.Tensor = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if i % 10 == 0:
            logging.info(f"Step {i} - loss: {loss.item()}")

    train_loss /= len(train_dataloader)
    logging.info(f"Train loss: {train_loss}")

    return train_loss


def validate_epoch(
    device: torch.device,
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    val_dataloader: DataLoader,
) -> float:
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_dataloader)
    logging.info(f"Validation loss: {val_loss}")

    return val_loss


def save_model(
    model_dir: str, model: nn.Module, full_dataset: datasets.ImageFolder
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": full_dataset.classes,
        },
        os.path.join(model_dir, "model.pt"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a model to predict the classification of a given image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing the training data.",
        required=True,
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Directory to save the trained model.",
        required=True,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to train the model.",
        default=10,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Number of images per batch.",
        default=32,
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        help="Number of folds for cross-validation.",
        default=5,
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logging.info(f"Training model with data from {args.data_dir}...")

    train(args.data_dir, args.model_dir, args.epochs, args.batch_size, args.n_splits)

    logging.info(f"Model saved to {args.model_dir}.")


if __name__ == "__main__":
    main()
