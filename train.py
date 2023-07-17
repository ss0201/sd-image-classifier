import argparse
import logging
import os
import sys
from typing import cast

import torch
import torch.utils.data
import torchvision


def train(data_dir, model_dir, epochs, batch_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}.")

    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(10),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    val_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    full_dataset = torchvision.datasets.ImageFolder(data_dir)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )
    train_dataset_image_folder = cast(
        torchvision.datasets.ImageFolder, train_dataset.dataset
    )
    val_dataset_image_folder = cast(
        torchvision.datasets.ImageFolder, val_dataset.dataset
    )
    train_dataset_image_folder.transform = train_transform
    val_dataset_image_folder.transform = val_transform

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, len(full_dataset.classes))
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10
    )

    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch + 1}...")
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
        scheduler.step(val_loss)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": full_dataset.classes,
        },
        os.path.join(model_dir, "model.pt"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train a model to predict the classfication of a given image."
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
