import argparse
import logging
import os
from typing import Tuple, cast

import torch
from coral_pytorch.dataset import corn_label_from_logits
from PIL import Image
from torch import nn
from torchvision.datasets.folder import is_image_file
from torchvision.transforms import Compose

from util import (
    TASK_CLASSIFICATION,
    TASK_ORDINAL_REGRESSION,
    create_model,
    get_device,
    get_val_transform,
)


def load_model(
    model_path: str, device: torch.device
) -> Tuple[nn.Module, list[str], int, str]:
    params = torch.load(model_path, map_location=device)
    model_state_dict = params["model_state_dict"]
    classes = params["classes"]
    resize_to = params["resize_to"]
    task_type = params["task_type"]
    model = create_model(device, len(classes), task_type)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model, classes, resize_to, task_type


def classify_images(
    data_dir: str,
    model: nn.Module,
    classes: list[str],
    resize_to: int,
    task_type: str,
    device: torch.device,
) -> None:
    transform = get_val_transform(resize_to)

    for file in os.listdir(data_dir):
        if not is_image_file(file):
            continue
        raw_image = Image.open(os.path.join(data_dir, file), mode="r").convert("RGB")
        class_name, likelihoods = predict_classification(
            raw_image, transform, model, classes, task_type, device
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
    task_type: str,
    device: torch.device,
) -> Tuple[str, torch.Tensor]:
    image = cast(torch.Tensor, transform(pil_image))
    image = torch.unsqueeze(image, 0)
    image = image.to(device)
    outputs: torch.Tensor = model(image)

    if task_type == TASK_CLASSIFICATION:
        _, predicted = torch.max(outputs.data, 1)
        class_name = classes[predicted[0]]
        likelihoods = nn.functional.softmax(outputs, dim=1)[0]
        return class_name, likelihoods
    elif task_type == TASK_ORDINAL_REGRESSION:
        predicted_labels = corn_label_from_logits(outputs).int()
        class_name = classes[predicted_labels[0]]

        rank_probabilities = torch.sigmoid(outputs)
        rank_probabilities = torch.cumprod(rank_probabilities, dim=1)
        rank_probabilities = torch.squeeze(rank_probabilities, dim=0)
        likelihoods = torch.cat(
            (torch.tensor([1.0], device=device), rank_probabilities), dim=0
        )

        return class_name, likelihoods
    else:
        raise ValueError(f"Unknown task type: {task_type}")


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
    model, classes, resize_to, task_type = load_model(args.model, device)
    classify_images(args.data_dir, model, classes, resize_to, task_type, device)


if __name__ == "__main__":
    main()
