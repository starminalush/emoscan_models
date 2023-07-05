"""Simple transforms for training."""

from torchvision import transforms


class BaselineImageTransform:
    def __init__(self, img_size=224):
        self.data_transform = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        (img_size - 4, img_size - 4), scale=(0.5, 1.0)
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.CenterCrop((img_size, img_size)),
                    transforms.ToTensor(),
                ]
            ),
        }

    def __call__(self, phase) -> transforms.Compose:
        return self.data_transform[phase]
