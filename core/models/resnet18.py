import torch
from torch import nn
from torchvision import models


class FERResnet18(nn.Module):
    def __init__(self, num_classes, device="cuda"):
        super().__init__()
        self.__model = self.__init_model(num_classes=num_classes)
        self.__model.to(device)

    @staticmethod
    def __init_model(num_classes):
        model = models.resnet18(pretrained=True)

        model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        model.fc = torch.nn.Linear(model.fc.in_features, num_classes, bias=True)

        return model

    def forward(self, x):
        return self.__model.forward(x)
