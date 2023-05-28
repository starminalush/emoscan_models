import torch

from core.trainers import Trainer


class DANTrainer(Trainer):
    def __init__(self, criterion_af, criterion_pt, **kwargs):
        super(DANTrainer, self).__init__(**kwargs)
        self.criterion_af = criterion_af
        self.criterion_pt = criterion_pt

    def _learn_on_batch(self, inputs, labels, phase):
        with torch.set_grad_enabled(phase == "train"):
            outputs, feat, heads = self._model_forward(inputs, phase=phase)
            loss: torch.Tensor = (
                self.criterion(outputs, labels)
                + self.criterion_af(feat, labels)
                + self.criterion_pt(heads)
            )

        return loss, outputs

    def _model_forward(self, inputs, **kwargs):
        outputs, feat, heads = self.model(inputs)
        if kwargs.get("phase") and kwargs.get("phase") in ["train", "val"]:
            return outputs, feat, heads
        else:
            return outputs
