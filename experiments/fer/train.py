import copy
import math
import os
from datetime import datetime
from pathlib import Path
from typing import List, OrderedDict, Tuple

import click
import mlflow
import torch
from dotenv import load_dotenv
from loguru import logger
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchmetrics import F1Score
from tqdm import tqdm
from utils import load_config, timeit

from core.data.datasets.fer2013 import FER2013Dataset
from core.data.transforms.baseline_transforms import ImageTransform
from core.models.resnet18 import FERResnet18

load_dotenv()


@timeit
def __train(
    num_epochs, dataloaders, device, optimizer, scheduler, model, criterion, f1_metric
) -> Tuple[nn.Module, float, float, List[float], List[float]]:
    """
    method for train model
    @return: model and training history
    @rtype:  Tuple(nn.Module, List, List, float, List, List)
    """
    best_model_wts: OrderedDict = copy.deepcopy(model.state_dict())
    best_loss: float = math.inf
    test_f1_history: List = []
    test_loss_history: List = []
    best_f1: float = 0.0

    for epoch in tqdm(range(num_epochs)):
        logger.debug(f"Epoch {epoch}/{num_epochs - 1}")
        logger.debug("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                scheduler.step()
                model.train()
            else:
                model.eval()

            current_loss: float = 0.0
            current_f1: float = 0.0

            for i, (inputs, labels) in tqdm(enumerate(dataloaders[phase])):
                inputs: torch.Tensor = inputs.to(device)
                labels: torch.Tensor = labels.to(device)

                optimizer.zero_grad()

                if i % 100 == 99:
                    logger.debug(
                        f"epoch {epoch}, loss {current_loss / (i * inputs.size(0))}"
                    )

                with torch.set_grad_enabled(phase == "train"):
                    outputs: torch.Tensor = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss: torch.Tensor = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                current_loss += loss.item() * inputs.size(0)
                current_f1 += f1_metric(preds, labels.data)

            epoch_loss: float = current_loss / len(dataloaders[phase].dataset)
            epoch_f1: float = current_f1 / len(dataloaders[phase].dataset)

            logger.info(f"phase {phase}, loss {epoch_loss}, acuuracy {epoch_f1}")

            if phase == "val":
                test_f1_history.append(epoch_f1)
                test_loss_history.append(epoch_loss)
                if epoch_loss < best_loss:
                    logger.debug("found best model")
                    logger.debug(
                        f"best model record loss: {epoch_loss}, previous record loss: {best_loss}"
                    )
                    best_loss = epoch_loss
                    best_f1 = epoch_f1
                    best_model_wts = copy.deepcopy(model.state_dict())

    logger.info(f"Best val f1: {best_f1:.4f} Best val loss: {best_loss:.4f}")

    model.load_state_dict(best_model_wts)
    return model, best_loss, best_f1, test_f1_history, test_loss_history


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("model-name", type=click.Path())
def train(config_path: str | Path, model_name: str | Path):
    experiment_id: str = mlflow.set_experiment("baseline_fer").experiment_id
    config = load_config(config_path=config_path)
    train_data = FER2013Dataset(
        config["datasets"]["train_dir"], transform=ImageTransform(), phase="train"
    )
    test_data = FER2013Dataset(
        config["datasets"]["val_dir"], transform=ImageTransform(), phase="val"
    )

    batch_size = config["datasets"]["batch_size"]

    dataloaders = {
        "train": DataLoader(train_data, batch_size=batch_size, pin_memory=True),
        "val": DataLoader(test_data, batch_size=batch_size, pin_memory=True),
    }

    model = FERResnet18(
        num_classes=len(train_data.classes), device=torch.device(os.getenv("DEVICE"))
    )
    max_of_counts_dataset = max(list(train_data.class_distribution.values()))
    classes_weights = list(
        map(
            lambda class_count: max_of_counts_dataset / class_count,
            list(train_data.class_distribution.values()),
        )
    )

    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.FloatTensor(classes_weights).to(os.getenv("DEVICE"))
    )
    optimizer = torch.optim.SGD(model.parameters(), **config["optimizer"])

    scheduler = StepLR(optimizer, **config["scheduler"])
    f1_metric = F1Score(task="multiclass", num_classes=len(train_data.classes))
    f1_metric.to(os.getenv('DEVICE'))
    with mlflow.start_run(experiment_id=experiment_id):
        model, best_loss, best_f1, test_f1_history, test_loss_history = __train(
            config["num_epochs"],
            dataloaders,
            os.getenv("DEVICE"),
            optimizer,
            scheduler,
            model,
            criterion,
            f1_metric,
        )
        Path(model_name).mkdir(exist_ok=True, parents=True)
        mlflow.pytorch.log_model(
            model,
            model_name,
            registered_model_name=model_name,
        )
        mlflow.log_artifact(config_path)
        mlflow.pytorch.save_model(model, model_name)
        mlflow.log_metric("Best F1", best_f1)
        mlflow.log_metric("Best loss", best_loss)
        for i in range(len(test_loss_history)):
            mlflow.log_metric("loss", test_loss_history[i])
            mlflow.log_metric("F1", test_f1_history[i])


if __name__ == "__main__":
    train()
