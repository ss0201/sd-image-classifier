import argparse
import logging
import os
import sys
from typing import Tuple, cast

import torch
import torch.utils.data
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


def train(data_dir: str, model_dir: str, epochs: int, batch_size: int) -> None:
    device = get_device()
    train_transform, val_transform = get_transforms()
    train_dataset, val_dataset, full_dataset = get_datasets(
        data_dir, train_transform, val_transform
    )
    train_dataloader, val_dataloader = get_dataloaders(
        train_dataset, val_dataset, batch_size
    )
    model = get_model(device, len(full_dataset.classes))
    criterion = get_criterion(device, full_dataset)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch + 1}...")
        train_epoch(device, model, criterion, optimizer, train_dataloader)
        val_loss = validate_epoch(device, model, criterion, val_dataloader)
        scheduler.step(val_loss)

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
    data_dir: str,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
) -> Tuple[Dataset, Dataset, datasets.ImageFolder]:
    full_dataset = datasets.ImageFolder(data_dir)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )
    train_dataset_image_folder = cast(datasets.ImageFolder, train_dataset.dataset)
    val_dataset_image_folder = cast(datasets.ImageFolder, val_dataset.dataset)
    train_dataset_image_folder.transform = train_transform
    val_dataset_image_folder.transform = val_transform

    return train_dataset, val_dataset, full_dataset


def get_dataloaders(
    train_dataset: Dataset, val_dataset: Dataset, batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader


def get_model(device: torch.device, num_classes: int) -> nn.Module:
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    return model


def get_criterion(
    device: torch.device, full_dataset: datasets.ImageFolder
) -> nn.CrossEntropyLoss:
    class_count = [0, 0, 0]
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
        if i % 100 == 0:
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
            "labels": full_dataset.classes,
        },
        os.path.join(model_dir, "model.pt"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a model to predict the classification of a given image."
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logging.info(f"Training model with data from {args.data_dir}...")

    train(args.data_dir, args.model_dir, args.epochs, args.batch_size)

    logging.info(f"Model saved to {args.model_dir}.")


if __name__ == "__main__":
    main()
