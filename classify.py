import argparse
import logging
import os
from typing import List, Tuple, cast

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_V2_M_Weights


def load_model(model_path: str, device: torch.device) -> Tuple[nn.Module, List[str]]:
    params = torch.load(model_path, map_location=device)
    model_state_dict = params["model_state_dict"]
    classes = params["labels"]
    model = models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, len(classes)),
    )
    model.load_state_dict(model_state_dict)
    model.eval()
    model = model.to(device)
    return model, classes


@torch.no_grad()
def classify(
    data_dir: str, model: nn.Module, classes: List[str], device: torch.device
) -> None:
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
        image_tensor = cast(torch.Tensor, transform(image))
        image_tensor = torch.unsqueeze(image_tensor, 0)

        image_tensor = image_tensor.to(device)
        outputs: torch.Tensor = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        class_name = classes[predicted[0]]
        likelihoods = nn.functional.softmax(outputs, dim=1)[0]
        logging.info(file)
        logging.info(f"  -> {class_name}")
        for i, label in enumerate(classes):
            logging.info(f"  {label}: {likelihoods[i]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify the data in the given directory using the trained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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

    model, classes = load_model(args.model, device)
    classify(args.data_dir, model, classes, device)


if __name__ == "__main__":
    main()
