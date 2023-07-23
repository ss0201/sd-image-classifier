import logging
import random
from typing import cast

import torch
import torch.utils.data
from PIL.Image import Image
from torch import nn
from torchvision import models, transforms
from torchvision.models import EfficientNet_V2_M_Weights
from torchvision.transforms import _functional_pil as F_pil
from torchvision.transforms import functional as F


def create_model(device: torch.device, num_classes: int) -> nn.Module:
    model = models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
    model.requires_grad_(False)

    lastconv_output_channels = cast(nn.Linear, model.classifier[1]).in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(lastconv_output_channels, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes),
    ).requires_grad_(True)

    return model.to(device)


def get_device() -> torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}.")
    return device


def pad_to_square(img: Image) -> Image:
    w, h = img.size
    if w == h:
        return img
    elif w < h:
        padding = (h - w) // 2
        return F_pil.pad(img, [padding, 0, padding, 0], fill=0)
    else:
        padding = (w - h) // 2
        return F_pil.pad(img, [0, padding, 0, padding], fill=0)


def random_crop_long_side(img: Image) -> Image:
    w, h = img.size
    if w < h:
        new_w = w
        new_h = int(h * random.triangular(0.9, 1, 1))
    else:
        new_w = int(w * random.triangular(0.9, 1, 1))
        new_h = h
    return F.center_crop(img, (new_h, new_w))  # type: ignore


def get_train_transform(resize_to: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(random_crop_long_side),
            transforms.Lambda(pad_to_square),
            transforms.Resize(resize_to),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_val_transform(resize_to: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Lambda(pad_to_square),
            transforms.Resize((resize_to, resize_to)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
