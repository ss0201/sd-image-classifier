"""
Train a model to predict the classfication of a given image.
Get the training data from the subdirectories of the given directory.
The subdirectory name is the label of the image.
Uses PyTorch to train the model.
"""

import argparse
import logging
import os
import sys

import torch
import torch.utils.data
import torchvision


def train(data_dir, model_dir, epochs, batch_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}.")

    train_dataset = torchvision.datasets.ImageFolder(data_dir)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset.transform = transform
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, len(train_dataset.classes))
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch + 1}...")
        model.train()
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                logging.info(f"Step {i} - loss: {loss.item()}")

    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))


def main():
    parser = argparse.ArgumentParser(
        description="Train a model to predict the classfication of a given image."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing the training data.",
        required=True,
    )
    parser.add_argument(
        "--model_dir",
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
        "--batch_size",
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
