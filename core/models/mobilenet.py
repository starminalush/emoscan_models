from torch import nn
from torchvision.models import mobilenet_v3_small

from core.models.base_model import BaseModel


class FERMobileNet(BaseModel):
    @staticmethod
    def _init_model(num_classes, pretrained) -> nn.Module:
        model: nn.Module = mobilenet_v3_small(pretrained)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features, num_classes, bias=True
        )
        return model
