import argparse
import copy
import logging
import os
import sys
from typing import Tuple, cast

import torch
import torch.utils.data
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from balanced_image_folder import BalancedImageFolder
from dataset_folder_subset import DatasetFolderSubset
from util import create_model, get_device, get_train_transform, get_val_transform


def train(
    data_dir: str,
    model_dir: str,
    resize_to: int,
    epochs: int,
    batch_size: int,
    n_splits: int,
    max_samples_per_class: int,
    oversample: bool,
    freeze_pretrained_layers: bool,
    device: torch.device,
) -> None:
    train_transform = get_train_transform(resize_to)
    val_transform = get_val_transform(resize_to)
    full_dataset = BalancedImageFolder(data_dir, max_samples_per_class, oversample)
    folds = create_datasets(full_dataset, n_splits, train_transform, val_transform)

    best_model = None
    best_val_loss = float("inf")
    val_losses = []

    for fold, (train_dataset, val_dataset) in enumerate(folds):
        logging.info(f"Starting fold {fold + 1}...")
        train_dataloader, val_dataloader = get_dataloaders(
            train_dataset, val_dataset, batch_size
        )

        model = create_model(
            device, len(full_dataset.classes), freeze_pretrained_layers
        )
        optimizer = get_optimizer(model)
        scheduler = get_scheduler(optimizer, epochs, len(train_dataloader))
        criterion = get_criterion(device, train_dataset.dataset_folder)

        for epoch in range(epochs):
            logging.info(f"Starting epoch {epoch + 1}...")
            train_epoch(device, model, criterion, optimizer, train_dataloader)
            val_loss = validate_epoch(device, model, criterion, val_dataloader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
            scheduler.step()

        val_losses.append(best_val_loss)
        logging.info(f"Finished fold {fold + 1}.")

    logging.info("Finished training.")

    avg_val_loss = sum(val_losses) / len(val_losses)
    logging.info(f"Average validation loss: {avg_val_loss}")
    for i, val_loss in enumerate(val_losses):
        logging.info(f"Fold {i + 1} validation loss: {val_loss}")

    if best_model is not None:
        save_model(model_dir, best_model, full_dataset.classes, resize_to)
    else:
        raise RuntimeError("No model was trained.")


def create_datasets(
    full_dataset: datasets.ImageFolder,
    n_splits: int,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
) -> list[Tuple[DatasetFolderSubset, DatasetFolderSubset]]:
    if n_splits > 1:
        folds = create_datasets_by_kfold(
            full_dataset, n_splits, train_transform, val_transform
        )
    else:
        train_dataset, val_dataset = create_datasets_by_holdout(
            full_dataset, train_transform, val_transform
        )
        folds = [(train_dataset, val_dataset)]

    return folds


def create_datasets_by_kfold(
    full_dataset: datasets.ImageFolder,
    n_splits: int,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
) -> list[Tuple[DatasetFolderSubset, DatasetFolderSubset]]:
    kfold = KFold(n_splits=n_splits)
    indices = [str(i) for i in range(len(full_dataset))]
    folds = []
    for train_indices, val_indices in kfold.split(indices):
        train_indices = cast(list[int], train_indices)
        val_indices = cast(list[int], val_indices)
        train_subset = DatasetFolderSubset(full_dataset, train_indices, train_transform)
        val_subset = DatasetFolderSubset(full_dataset, val_indices, val_transform)
        folds.append((train_subset, val_subset))

    return folds


def create_datasets_by_holdout(
    full_dataset: datasets.ImageFolder,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
) -> Tuple[DatasetFolderSubset, DatasetFolderSubset]:
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    train_subset = DatasetFolderSubset(
        full_dataset, list(train_subset.indices), train_transform
    )
    val_subset = DatasetFolderSubset(
        full_dataset, list(val_subset.indices), val_transform
    )
    return train_subset, val_subset


def get_dataloaders(
    train_dataset: DatasetFolderSubset,
    val_dataset: DatasetFolderSubset,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader


def get_criterion(
    device: torch.device, dataset: datasets.DatasetFolder
) -> nn.CrossEntropyLoss:
    class_count = [0] * len(dataset.classes)
    for _, class_idx in dataset.samples:
        class_count[class_idx] += 1
    class_weights = 1.0 / torch.tensor(class_count, dtype=torch.float)
    class_weights_normalized = (class_weights / class_weights.sum()).to(device)

    return nn.CrossEntropyLoss(weight=class_weights_normalized)


def get_optimizer(model: nn.Module) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=0.001)


def get_scheduler(
    optimizer: optim.Optimizer, epochs: int, steps_per_epoch: int
) -> LRScheduler:
    return OneCycleLR(
        optimizer, max_lr=0.1, epochs=epochs, steps_per_epoch=steps_per_epoch
    )


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
    model_dir: str, model: nn.Module, classes: list[str], resize_to: int
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "resize_to": resize_to,
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
        "--resize-to",
        type=int,
        help="Size to resize the images to.",
        default=480,
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
        help="Number of folds for cross-validation. If 1, holdout validation is used.",
        default=1,
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        help="Maximum number of samples per class to use for training. "
        "If 0, all samples are used.",
        default=0,
    )
    parser.add_argument(
        "--oversample",
        action="store_true",
        help="Oversample the dataset to balance the classes.",
    )
    parser.add_argument(
        "--unfreeze-pretrained",
        action="store_true",
        help="Unfreeze the weights of the pretrained layers.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    device = get_device()

    logging.info(f"Training model with data from {args.data_dir}...")
    train(
        args.data_dir,
        args.model_dir,
        args.resize_to,
        args.epochs,
        args.batch_size,
        args.n_splits,
        args.max_samples_per_class,
        args.oversample,
        not args.unfreeze_pretrained,
        device,
    )

    logging.info(f"Model saved to {args.model_dir}.")


if __name__ == "__main__":
    main()
