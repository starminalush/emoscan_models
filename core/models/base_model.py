from abc import ABC, abstractstaticmethod

from torch import Tensor, nn


class BaseModel(ABC, nn.Module):
    def __init__(self, num_class, pretrained=False):
        super().__init__()
        self._model = self._init_model(num_classes=num_class, pretrained=pretrained)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Get output from model.

        Args:
            input_tensor: Input tensor.

        Returns:
            Output tensor.
        """
        return self._model.forward(input_tensor)

    @staticmethod(abstractstaticmethod)
    def _init_model(num_classes, pretrained) -> nn.Module:
        raise NotImplementedError
