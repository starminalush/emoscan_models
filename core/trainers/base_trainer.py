import math
import time
from typing import Dict, List, Tuple

import torch
from loguru import logger
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        dataloaders: Dict[str, DataLoader],
        device: torch.device,
        optimizer: Optimizer,
        criterion: Module,
        scheduler: LRScheduler,
        model: Module,
        metrics: Metric,
        per_class_metrics: Metric,
    ):
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.per_class_metrics = per_class_metrics
        self.model = model
        self.scheduler = scheduler
        self.dataloaders = dataloaders
        self.device = device

    def train(
        self, num_epochs: int
    ) -> Tuple[float, float, Module, List[float], List[float]]:
        """Train model for number of epochs
        Args:
            num_epochs: number of epochs

        Returns:
            A tuple containing best model, loss and metric, and also train history

        """
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
        """Starts training for one epoch.
        Returns:
            A tuple containing loss and metric
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
                loss, outputs = self._learn_step(inputs, labels, phase)
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

    def calculate_throughput(self) -> float:
        dummy_input = torch.randn(5, 3, 224, 224, dtype=torch.float).to(self.device)
        repetitions = 100
        total_time = 0
        with torch.no_grad():
            for rep in range(repetitions):
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                starter.record()
                self.model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender) / 1000
                total_time += curr_time
        throughput = repetitions * 5 / total_time
        return throughput

    def calculate_latency(self) -> float:
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float).to(self.device)
        start = time.time()
        self.model(dummy_input)
        end = time.time()
        latency = end - start
        return latency

    def test(self) -> Tuple[float, float]:
        """Starts testing the best model on test subset
        Returns:
            A tuple containing loss and metric
        """
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

    def _learn_step(self, inputs, labels, phase) -> Tuple[Tensor, Tensor]:
        with torch.set_grad_enabled(phase == "train"):
            outputs: Tensor = self._model_forward(inputs)
            loss: Tensor = self.criterion(outputs, labels)
        return loss, outputs

    def _model_forward(self, inputs) -> Tensor:
        return self.model(inputs)
