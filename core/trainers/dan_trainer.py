import math
from typing import List, Tuple

import torch
from loguru import logger
from torch import Tensor, nn
from tqdm import tqdm

import mlflow
from core.trainers import Trainer


class DANTrainer(Trainer):
    def __init__(self, criterion_af, criterion_pt, **kwargs):
        super(DANTrainer, self).__init__(**kwargs)
        self.criterion_af = criterion_af
        self.criterion_pt = criterion_pt

    def _learn_on_batch(self, inputs, labels, phase):
        with torch.set_grad_enabled(phase == "train"):
            outputs, feat, heads = self._model_forward(inputs, phase=phase)
            loss_cls = self.criterion(outputs, labels)
            loss_pt = self.criterion_pt(heads)
            loss_af = self.criterion_af(feat, labels)
            loss: torch.Tensor = (
                self.criterion(outputs, labels)
                + self.criterion_af(feat, labels)
                + self.criterion_pt(heads)
            )

        return loss, outputs, loss_cls, loss_pt, loss_af

    def _train_one_epoch(self) -> Tuple[float, float, float, float, float]:
        """Запускает обучение на одной эпохе.
        Returns:

        """
        for phase in ["train", "val"]:
            if phase == "train":
                self.scheduler.step()
                self.model.train()
            else:
                self.model.eval()

            current_loss, current_loss_cls, current_loss_pt, current_loss_af = (
                0.0,
                0.0,
                0.0,
                0.0,
            )

            for idx, (inputs, labels) in tqdm(enumerate(self.dataloaders[phase])):
                self.optimizer.zero_grad()
                inputs: Tensor = inputs.to(self.device)
                labels: Tensor = labels.to(self.device)
                loss, outputs, loss_cls, loss_pt, loss_af = self._learn_on_batch(
                    inputs, labels, phase
                )
                _, preds = torch.max(outputs, 1)
                if phase == "train":
                    loss_cls.backward(retain_graph=True)
                    loss_pt.backward(retain_graph=True)
                    loss_af.backward(retain_graph=True)
                    loss.backward()
                    self.optimizer.step()
                current_loss += loss.item()
                current_loss_cls += loss_cls.item()
                current_loss_pt += loss_pt.item()
                current_loss_af += loss_af.item()
                if phase == "val":
                    self.metrics.update(preds, labels)

        epoch_loss: float = current_loss / (idx + 1)
        epoch_loss_cls: float = current_loss_cls / (idx + 1)
        epoch_loss_af: float = current_loss_af / (idx + 1)
        epoch_loss_pt: float = current_loss_pt / (idx + 1)
        epoch_metrics: float = self.metrics.compute()
        self.metrics.reset()

        return epoch_metrics, epoch_loss, epoch_loss_cls, epoch_loss_af, epoch_loss_pt

    def train(
        self, num_epochs
    ) -> Tuple[
        float,
        float,
        nn.Module,
        List[float],
        List[float],
        List[float],
        List[float],
        List[float],
    ]:
        best_loss: float = math.inf
        best_metrics: float = 0.0
        best_model = None
        test_f1_history: List[float] = []
        test_loss_history: List[float] = []
        test_loss_history_cls: List[float] = []
        test_loss_history_pt: List[float] = []
        test_loss_history_af: List[float] = []

        for idx in range(num_epochs):
            metrics, loss, loss_cls, loss_af, loss_pt = self._train_one_epoch()
            test_f1_history.append(metrics)
            test_loss_history.append(loss)
            test_loss_history_cls.append(loss_cls)
            test_loss_history_pt.append(loss_pt)
            test_loss_history_af.append(loss_af)
            logger.info(f"Epoch {idx}/{num_epochs} loss {loss}, f1 {metrics}")
            logger.info("-" * 10)
            if loss < best_loss:
                best_loss = loss
                best_metrics = metrics
                best_model = self.model
        return (
            best_loss,
            best_metrics,
            best_model,
            test_loss_history,
            test_f1_history,
            test_loss_history_cls,
            test_loss_history_af,
            test_loss_history_pt,
        )

    def _model_forward(self, inputs, **kwargs):
        outputs, feat, heads = self.model(inputs)
        if kwargs.get("phase") and kwargs.get("phase") in ["train", "val"]:
            return outputs, feat, heads
        else:
            return outputs
