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
        self.classes = sorted([f for f in self.dataset_path.iterdir() if f.is_dir()])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        # Transformimg Image
        if self.transform:
            img = self.transform(img, self.phase)
        # Get Label
        label = img_path.parent
        label_idx = torch.tensor(self.classes.index(label)).type(torch.long)

        return img, label_idx

    @property
    def class_distribution(self):
        return {
            self.classes.index(f): len([*f.iterdir()])
            for f in self.dataset_path.rglob("*")
            if f.is_dir()
        }
