import os
from pathlib import Path
from typing import Dict, List, Tuple

import click
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torchsampler import ImbalancedDatasetSampler
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose
from utils import download_pretrained_model_from_gdrive, init_module, load_config, mlflow_logger

from core.trainers import Trainer


@click.command()
@click.argument("config-path", type=click.Path(exists=True, path_type=Path))
@click.argument("model-output-path", type=click.Path(path_type=Path))
@click.argument("checkpoint-path", type=click.Path(path_type=Path), required=False)
@mlflow_logger("fer")
def train(
    config_path: Path | str,
    model_output_path: Path | str,
    checkpoint_path: Path | str = None,
) -> Tuple[
    Module,
    float,
    float,
    List[float],
    List[float],
    float,
    Dict[str, float],
    float,
    float,
]:
    """Start training the model based on the config.

    Args:
        config_path: Model and params config.
        model_output_path: Path of trained model.
        checkpoint_path: Checkpoint path for initial model.

    Returns:
        A tuple containing best trained model, loss and metrics, train history, throughout and latency of best model.

    """
    config: Dict[str, str | float | int] = load_config(config_path=config_path)

    transforms: Compose = init_module(config["transforms"]["class_str"])(
        **config["transforms"]["params"]
    )
    train_data: VisionDataset = init_module(config["dataset"]["class_str"])(
        config["dataset_path"]["train_dir"], transform=transforms(phase="train")
    )
    val_data: VisionDataset = init_module(config["dataset"]["class_str"])(
        config["dataset_path"]["val_dir"], transform=transforms(phase="val")
    )
    # все равно у val и test выборки одинаковые аугментации
    test_data: VisionDataset = init_module(config["dataset"]["class_str"])(
        config["dataset_path"]["test_dir"], transform=transforms(phase="val")
    )
    batch_size: int = int(config["batch_size"])
    num_epochs: int = config["num_epochs"]

    dataloaders: Dict[str, DataLoader] = {
        "train": DataLoader(
            train_data,
            batch_size=batch_size,
            pin_memory=True,
            sampler=ImbalancedDatasetSampler(train_data),
        ),
        "val": DataLoader(val_data, batch_size=batch_size, pin_memory=True),
        "test": DataLoader(test_data, batch_size=batch_size, pin_memory=True),
    }

    device: torch.device = torch.device(os.getenv("DEVICE"))

    model: Module = init_module(config["model"]["class_str"])(
        **config["model"]["params"]
    )
    if checkpoint_path:
        if not checkpoint_path.exists():
            download_pretrained_model_from_gdrive(
                file_id=config["model"]["checkpoint_gdrive_id"],
                output_model_name=str(checkpoint_path),
            )
        checkpoint = torch.load(
            f=checkpoint_path,
            map_location=device,
        )
        model.load_state_dict(state_dict=checkpoint["model_state_dict"], strict=True)
    model.to(device)

    criterion_cls: Module = init_module(config["criterion_cls"]["class_str"])()
    criterion_af: Module = init_module(config["criterion_af"]["class_str"])(
        **config["criterion_af"]["params"], device=device
    )
    criterion_pt: Module = init_module(config["criterion_pt"]["class_str"])()
    metric: Metric = init_module(config["metrics"]["class_str"])(
        **config["metrics"]["params"]
    )
    metric.to(device)

    per_class_metric: Metric = init_module(config["metrics"]["class_str"])(
        **config["metrics"]["params"], average=None
    )
    per_class_metric.to(device)

    optimizer_params: List = list(model.parameters()) + list(criterion_af.parameters())
    optimizer: Optimizer = init_module(config["optimizer"]["class_str"])(
        optimizer_params, **config["optimizer"]["params"]
    )
    scheduler: LRScheduler = init_module(config["scheduler"]["class_str"])(
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
        criterion_pt=criterion_pt,
        criterion_af=criterion_af,
    )

    # train and valid
    (
        best_loss,
        best_metrics,
        best_model,
        val_loss_history,
        val_metrics_history,
    ) = trainer.train(num_epochs=num_epochs)
    torch.save(obj=best_model, f=model_output_path)

    # test
    test_metric, per_class_metrics = trainer.test()
    inv_map: Dict[int, float] = {cls_index: cls_label for cls_label, cls_index in train_data.class_to_idx.items()}
    per_class_metrics: Dict[str, float] = {
        inv_map[idx]: metric_value for idx, metric_value in enumerate(per_class_metrics)
    }
    throughput: float = trainer.calculate_throughput()
    latency: float = trainer.calculate_latency()
    return (
        best_model,
        best_loss,
        best_metrics,
        val_loss_history,
        val_metrics_history,
        test_metric,
        per_class_metrics,
        throughput,
        latency,
    )


if __name__ == "__main__":
    train()
