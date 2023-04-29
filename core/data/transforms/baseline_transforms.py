from torchvision import transforms


class ImageTransform:
    def __init__(self, img_size=224):
        self.data_transform = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        (img_size, img_size), scale=(0.5, 1.0)
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.CenterCrop((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)
