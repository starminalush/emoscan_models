import os
from pathlib import Path
from typing import Dict

import click
import torch
from dotenv import load_dotenv
from torch import nn
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
from utils import init_module, load_config, mlflow_logger

from core.trainers import Trainer

load_dotenv()


@click.command()
@click.argument("config-path", type=click.Path(exists=True))
@click.argument("model-output-path", type=click.Path())
@mlflow_logger("fer")
def train(config_path: str | Path, model_output_path: str | Path):
    config: Dict[str, str | float | int] = load_config(config_path=config_path)

    transforms = init_module(config["transforms"]["class_str"])(
        **config["transforms"]["params"]
    )
    train_data = init_module(config["dataset"]["class_str"])(
        config["dataset_path"]["train_dir"], transform=transforms(phase="train")
    )
    val_data = init_module(config["dataset"]["class_str"])(
        config["dataset_path"]["val_dir"], transform=transforms(phase="val")
    )
    # все равно у val и test выборки одинаковые аугментации
    test_data = init_module(config["dataset"]["class_str"])(
        config["dataset_path"]["val_dir"], transform=transforms(phase="val")
    )
    batch_size = int(config["batch_size"])
    num_epochs = config["num_epochs"]

    dataloaders = {
        "train": DataLoader(
            train_data,
            batch_size=batch_size,
            pin_memory=True,
            sampler=ImbalancedDatasetSampler(train_data),
        ),
        "val": DataLoader(val_data, batch_size=batch_size, pin_memory=True),
        "test": DataLoader(test_data, batch_size=batch_size, pin_memory=True),
    }

    device = torch.device(os.getenv("DEVICE"))
    model: nn.Module = init_module(config["model"]["class_str"])(
        **config["model"]["params"]
    )
    model.to(device)

    criterion_cls = init_module(config["criterion_cls"]["class_str"])()
    metric = init_module(config["metrics"]["class_str"])(**config["metrics"]["params"])
    metric.to(device)

    per_class_metric = init_module(config["metrics"]["class_str"])(
        **config["metrics"]["params"], average=None
    )
    per_class_metric.to(device)

    # params = list(model.parameters()) + list(criterion_af.parameters())
    optimizer = init_module(config["optimizer"]["class_str"])(
        model.parameters(), **config["optimizer"]["params"]
    )
    scheduler = init_module(config["scheduler"]["class_str"])(
        optimizer, **config["scheduler"]["params"]
    )
    trainer: Trainer = init_module(config["trainer"]["class_str"])(
        dataloaders=dataloaders,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion_cls,
        model=model,
        metrics=metric,
        per_class_metrics=per_class_metric,
    )

    # train and valid
    (
        best_loss,
        best_metrics,
        best_model,
        val_loss_history,
        val_metrics_history,
    ) = trainer.train(num_epochs=num_epochs)
    torch.save(best_model, model_output_path)

    # test
    test_metric, per_class_metrics = trainer.test()
    inv_map = {v: k for k, v in train_data.class_to_idx.items()}
    per_class_metrics = {
        inv_map[idx]: value for idx, value in enumerate(per_class_metrics)
    }
    return (
        best_model,
        best_loss,
        best_metrics,
        val_loss_history,
        val_metrics_history,
        test_metric,
        per_class_metrics,
    )


if __name__ == "__main__":
    train()
