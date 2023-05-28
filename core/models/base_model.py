from abc import ABC, abstractstaticmethod

from torch import Tensor, nn


class BaseModel(ABC, nn.Module):
    def __init__(self, num_class, pretrained=False):
        super().__init__()
        self.__model = self._init_model(num_classes=num_class, pretrained=pretrained)

    @staticmethod(abstractstaticmethod)
    def _init_model(num_classes, pretrained) -> nn.Module:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        return self.__model.forward(x)
