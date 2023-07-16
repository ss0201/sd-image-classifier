import argparse
import logging
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torchvision import models


def load_model(model_path, device):
    params = torch.load(model_path, map_location=device)
    model_state_dict = params["model_state_dict"]
    labels = params["labels"]
    model = models.resnet18(pretrained=True).to(device)
    model.fc = nn.Linear(512, len(labels)).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model, labels


def classify(data_dir, model, labels, device):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    for file in os.listdir(data_dir):
        image = Image.open(os.path.join(data_dir, file))
        image = transform(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            image = image.to(device)
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            label_name = labels[predicted[0]]

            logging.info(f"{label_name} - {file}")


def main():
    parser = argparse.ArgumentParser(
        description="Classify the data in the given directory using the trained model."
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        help="The directory containing the data to classify.",
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="The model to use for classification.",
        required=True,
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}.")

    model, labels = load_model(args.model, device)
    classify(args.data_dir, model, labels, device)


if __name__ == "__main__":
    main()
