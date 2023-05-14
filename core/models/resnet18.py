import torch
from torch import nn
from torchvision import models


class FERResnet18(nn.Module):
    def __init__(self, num_class, pretrained="False"):
        super().__init__()
        self.__model = self.__init_model(num_classes=num_class, pretrained=pretrained)

    @staticmethod
    def __init_model(num_classes, pretrained):
        model = models.resnet18(pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes, bias=True)

        return model

    def forward(self, x):
        return self.__model.forward(x)
