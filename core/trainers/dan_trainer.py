import torch
from torch import Tensor
from torch.nn import Module

from core.trainers import Trainer


class DANTrainer(Trainer):
    def __init__(self, criterion_af: Module, criterion_pt: Module, **kwargs):
        super().__init__(**kwargs)
        self.criterion_af: Module = criterion_af
        self.criterion_pt: Module = criterion_pt

    def _learn_step(self, inputs: Tensor, labels: Tensor, phase: str):
        with torch.set_grad_enabled(phase == "train"):
            outputs, feat, heads = self._model_forward(inputs, phase=phase)
            loss: torch.Tensor = (
                self.criterion(outputs, labels) + self.criterion_af(feat, labels) + self.criterion_pt(heads)
            )

        return loss, outputs

    def _model_forward(self, inputs: Tensor, **kwargs):
        outputs, feat, heads = self.model(inputs)
        if kwargs.get("phase") and kwargs.get("phase") in {"train", "val"}:
            return outputs, feat, heads
        return outputs
