from torch import nn
from torchvision.models import resnet18

from core.models.base_model import BaseModel


class FERResnet18(BaseModel):
    @staticmethod
    def _init_model(num_classes, pretrained) -> nn.Module:
        model: nn.Module = resnet18(pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes, bias=True)
        return model
