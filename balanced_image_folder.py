import random
from collections import Counter
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader


class BalancedImageFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        max_samples_per_class: int,
        oversample: bool,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.max_samples_per_class = max_samples_per_class
        self.will_oversample = oversample
        super().__init__(
            root, transform, target_transform, loader, is_valid_file=is_valid_file
        )

    def make_dataset(
        self,
        directory: str,
        class_to_idx: dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> list[Tuple[str, int]]:
        samples = super().make_dataset(
            directory, class_to_idx, extensions, is_valid_file
        )
        if self.max_samples_per_class > 0:
            samples = self.undersample(samples, self.max_samples_per_class)
        if self.will_oversample:
            samples = self.oversample(samples)

        return samples

    def undersample(
        self, samples: list[Tuple[str, int]], max_samples_per_class: int
    ) -> list[Tuple[str, int]]:
        undersampled_samples = []
        class_count = Counter([sample[1] for sample in samples])
        for _class in class_count:
            class_samples = [sample for sample in samples if sample[1] == _class]
            if len(class_samples) > max_samples_per_class:
                class_samples = random.sample(class_samples, max_samples_per_class)
            undersampled_samples.extend(class_samples)

        return undersampled_samples

    def oversample(self, samples: list[Tuple[str, int]]) -> list[Tuple[str, int]]:
        oversampled_samples = samples.copy()
        class_count = Counter([sample[1] for sample in samples])
        max_class = max(class_count, key=lambda x: class_count[x])
        for _class in class_count:
            num_samples_to_add = class_count[max_class] - class_count[_class]
            class_samples = [sample for sample in samples if sample[1] == _class]
            for _ in range(num_samples_to_add):
                sample = random.choice(class_samples)
                oversampled_samples.append(sample)

        return oversampled_samples
