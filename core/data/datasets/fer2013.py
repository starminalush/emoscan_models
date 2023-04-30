from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset


class FER2013Dataset(Dataset):
    def __init__(self, dataset_path: Path, transform=None, phase="train"):
        self.dataset_path: Path = Path(dataset_path)
        self.file_list = [f for f in self.dataset_path.rglob("*.jpg") if f.is_file()]
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        # Transformimg Image
        if self.transform:
            img = self.transform(img, self.phase)
        # Get Label
        label = img_path.parent.name
        label_idx = -1
        match label:
            case "happy":
                label_idx = 0
            case "neutral":
                label_idx = 1
            case _:
                label_idx = 2

        label_idx = torch.tensor(label_idx).type(torch.long)
        return img, label_idx

    @property
    def class_distribution(self):
        result_dict = {
            0: len([f for f in (self.dataset_path / "happy").iterdir()]),
            1: len([f for f in (self.dataset_path / "neutral").iterdir()]),
        }
        result_dict[2] = (
            len([f for f in self.dataset_path.rglob("*.jpg")])
            - result_dict[0]
            - result_dict[1]
        )
        return result_dict
