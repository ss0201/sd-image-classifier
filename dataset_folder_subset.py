from typing import Callable, Optional

from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder

from util import DatasetFolderItem


class DatasetFolderSubset(Dataset[DatasetFolderItem]):
    dataset_folder: DatasetFolder
    indices: list[int]

    def __init__(
        self,
        dataset: DatasetFolder,
        indices: list[int],
        transform: Optional[Callable] = None,
    ):
        self.dataset_folder = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, index: int) -> DatasetFolderItem:
        sample, target = self.dataset_folder[self.indices[index]]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]
