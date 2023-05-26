from typing import Tuple

import torch
from loguru import logger
from torch import Tensor, nn

from core.trainers import Trainer


class DANTrainer(Trainer):
    def __init__(
        self,
        dataloaders,
        device,
        optimizer,
        criterion,
        scheduler,
        model,
        metrics,
        per_class_metrics,
        criterion_af,
        criterion_pt,
    ):
        super(DANTrainer, self).__init__(
            dataloaders,
            device,
            optimizer,
            criterion,
            scheduler,
            model,
            metrics,
            per_class_metrics,
        )
        self.criterion_af = criterion_af
        self.criterion_pt = criterion_pt

    def _learn_on_batch(self, inputs, labels, phase):
        with torch.set_grad_enabled(phase == "train"):
            outputs, feat, heads = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss: Tensor = (
                self.criterion(outputs, labels)
                + self.criterion_af(feat, labels)
                + self.criterion_pt(heads)
            )

            if phase == "train":
                loss.backward()
                self.optimizer.step()

        return loss, preds

    def test(self) -> Tuple[float, float]:
        with torch.no_grad():
            for data in self.dataloaders["test"]:
                images, labels = data
                images: torch.Tensor = images.to(self.device)
                labels: torch.Tensor = labels.to(self.device)
                outputs, _, _ = self.model(images)
                _, predicted = torch.max(outputs, 1)
                self.metrics.update(labels, predicted)
                self.per_class_metrics.update(labels, predicted)

        current_metric = self.metrics.compute()
        per_class_metric = self.per_class_metrics.compute()
        self.metrics.reset()
        self.per_class_metrics.reset()

        return current_metric, per_class_metric
