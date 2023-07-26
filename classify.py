import argparse
import logging
import os
from typing import Tuple, cast

import torch
from PIL import Image
from torch import nn
from torchvision.transforms import Compose

from util import create_model, get_device, get_val_transform


def load_model(
    model_path: str, device: torch.device
) -> Tuple[nn.Module, list[str], int]:
    params = torch.load(model_path, map_location=device)
    model_state_dict = params["model_state_dict"]
    classes = params["classes"]
    resize_to = params["resize_to"]
    model = create_model(device, len(classes))
    model.load_state_dict(model_state_dict)
    model.eval()
    return model, classes, resize_to


def classify_images(
    data_dir: str,
    model: nn.Module,
    classes: list[str],
    resize_to: int,
    device: torch.device,
) -> None:
    transform = get_val_transform(resize_to)

    for file in os.listdir(data_dir):
        raw_image = Image.open(os.path.join(data_dir, file), mode="r").convert("RGB")
        class_name, likelihoods = predict_classification(
            raw_image, transform, model, classes, device
        )
        logging.info(file)
        logging.info(f"  -> {class_name}")
        for i, cls in enumerate(classes):
            logging.info(f"  {cls}: {likelihoods[i]:.4f}")


@torch.no_grad()
def predict_classification(
    pil_image: Image.Image,
    transform: Compose,
    model: nn.Module,
    classes: list[str],
    device: torch.device,
) -> Tuple[str, torch.Tensor]:
    image = cast(torch.Tensor, transform(pil_image))
    image = torch.unsqueeze(image, 0)
    image = image.to(device)
    outputs: torch.Tensor = model(image)
    _, predicted = torch.max(outputs.data, 1)
    class_name = classes[predicted[0]]
    likelihoods = nn.functional.softmax(outputs, dim=1)[0]
    return class_name, likelihoods


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

    device = get_device()
    model, classes, resize_to = load_model(args.model, device)
    classify_images(args.data_dir, model, classes, resize_to, device)


if __name__ == "__main__":
    main()
