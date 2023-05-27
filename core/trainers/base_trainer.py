import math
from typing import List, Tuple

import torch
from loguru import logger
from torch import Tensor, nn
from tqdm import tqdm


class Trainer:
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
    ):
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.per_class_metrics = per_class_metrics
        self.model: nn.Module = model
        self.scheduler = scheduler
        self.dataloaders = dataloaders
        self.device = device

    def train(
        self, num_epochs
    ) -> Tuple[float, float, nn.Module, List[float], List[float]]:
        best_loss: float = math.inf
        best_metrics: float = 0.0
        best_model = None
        test_f1_history: List[float] = []
        test_loss_history: List[float] = []
        for idx in range(num_epochs):
            metrics, loss = self._train_one_epoch()
            test_f1_history.append(metrics)
            test_loss_history.append(loss)
            logger.info(f"Epoch {idx}/{num_epochs} loss {loss}, f1 {metrics}")
            logger.info("-" * 10)
            if loss < best_loss:
                best_loss = loss
                best_metrics = metrics
                best_model = self.model
        return best_loss, best_metrics, best_model, test_loss_history, test_f1_history

    def _train_one_epoch(self) -> Tuple[float, float]:
        """Запускает обучение на одной эпохе.
        Returns:

        """
        for phase in ["train", "val"]:
            if phase == "train":
                self.scheduler.step()
                self.model.train()
            else:
                self.model.eval()

            current_loss: float = 0.0

            for idx, (inputs, labels) in tqdm(enumerate(self.dataloaders[phase])):
                self.optimizer.zero_grad()
                inputs: Tensor = inputs.to(self.device)
                labels: Tensor = labels.to(self.device)
                loss, outputs = self._learn_on_batch(inputs, labels, phase)
                _, preds = torch.max(outputs, 1)
                if phase == "train":
                    loss.backward()
                    self.optimizer.step()
                current_loss += loss.item()
                if phase == "val":
                    self.metrics.update(preds, labels)

        epoch_loss: float = current_loss / (idx + 1)
        epoch_metrics: float = self.metrics.compute()
        self.metrics.reset()

        return epoch_metrics, epoch_loss

    def calculate_throughtput(self):
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(self.device)
        repetitions = 100
        total_time = 0
        with torch.no_grad():
            for rep in range(repetitions):
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                starter.record()
                _ = self.model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender) / 1000
                total_time += curr_time
        throughput = repetitions / total_time
        return throughput

    def test(self) -> Tuple[float, float]:
        with torch.no_grad():
            for data in self.dataloaders["test"]:
                images, labels = data
                images: torch.Tensor = images.to(self.device)
                labels: torch.Tensor = labels.to(self.device)
                outputs: torch.Tensor = self._model_forward(images)
                _, predicted = torch.max(outputs, 1)
                self.metrics.update(labels, predicted)
                self.per_class_metrics.update(labels, predicted)

        current_metric = self.metrics.compute()
        per_class_metric = self.per_class_metrics.compute()
        self.metrics.reset()
        self.per_class_metrics.reset()

        return current_metric, per_class_metric

    def _learn_on_batch(self, inputs, labels, phase):
        with torch.set_grad_enabled(phase == "train"):
            outputs: Tensor = self._model_forward(inputs)
            loss: Tensor = self.criterion(outputs, labels)
        return loss, outputs

    def _model_forward(self, inputs):
        return self.model(inputs)
