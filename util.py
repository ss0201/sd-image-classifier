import logging
from typing import cast

import torch
import torch.utils.data
from torch import nn
from torchvision import models, transforms
from torchvision.models import EfficientNet_V2_M_Weights


def create_model(device: torch.device, num_classes: int) -> nn.Module:
    model = models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
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
    )
    return model.to(device)


def get_device() -> torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}.")
    return device


def get_train_transform(resize_to: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize((resize_to, resize_to)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_val_transform(resize_to: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((resize_to, resize_to)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
