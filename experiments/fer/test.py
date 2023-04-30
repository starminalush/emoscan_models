import os
from pathlib import Path

import click
import torch
from dotenv import load_dotenv
from loguru import logger
from torch.utils.data import DataLoader
from torchmetrics import F1Score
from utils import load_config

import mlflow
from core.data.datasets.fer2013 import FER2013Dataset
from core.data.transforms.baseline_transforms import ImageTransform

load_dotenv(".env")


def __test(model, test_dataloader, f1_metric, device):
    current_f1 = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images: torch.Tensor = images.to(device)
            labels: torch.Tensor = labels.to(device)
            outputs: torch.Tensor = model(images)
            _, predicted = torch.max(outputs, 1)
            current_f1 += f1_metric(labels, predicted)

    current_f1 /= len(test_dataloader)
    return current_f1


@click.command()
@click.argument("config-path", type=click.Path(exists=True))
@click.argument("model-path", type=click.Path(exists=True))
def test(config_path: str | Path, model_path: str | Path):
    experiment_id: str = mlflow.set_experiment("baseline_fer").experiment_id
    model = torch.load(model_path)
    model.to(os.getenv("DEVICE"))

    config = load_config(config_path=config_path)

    test_data = FER2013Dataset(
        config["datasets"]["train_dir"], transform=ImageTransform(), phase="val"
    )
    test_dataloader = DataLoader(
        test_data, batch_size=config["datasets"]["batch_size"], pin_memory=True
    )

    f1_metric = F1Score(task="multiclass", num_classes=3, average=None)
    f1_metric.to(os.getenv("DEVICE"))

    f1_test = __test(model, test_dataloader, f1_metric, os.getenv("DEVICE"))
    logger.error(f1_test)
    with mlflow.start_run(experiment_id=experiment_id):
        for idx, f1 in enumerate(f1_test):
            mlflow.log_metric(f"Test F1_class_{idx}", f1)


if __name__ == "__main__":
    test()
