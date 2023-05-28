import functools
import importlib
from pathlib import Path
from time import time

import gdown
import yaml
from loguru import logger

import mlflow


def download_pretrained_model_from_gdrive(file_id: str, output_model_name: Path | str):
    """Download model from gdrive based on fileid
    Args:
        file_id: file id in gdrive
        output_model_name: output_file_name
    """
    uri = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(uri, output_model_name, quiet=False)


def load_config(config_path: str):
    with open(config_path) as src:
        config: dict = yaml.load(src, Loader=yaml.Loader)
    return config


def init_module(class_str: str):
    """Initialize class by import name

    Args:
        class_str: import path of class
    """
    module_path, class_name = class_str.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def mlflow_logger(experiment_name: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            experiment_id: str = mlflow.set_experiment(experiment_name).experiment_id
            with mlflow.start_run(experiment_id=experiment_id):
                (
                    model,
                    best_loss,
                    best_metrics,
                    val_loss_history,
                    val_metrics_history,
                    test_metric,
                    per_class_metrics,
                    throughtput,
                    latency,
                ) = func(*args, **kwargs)
                mlflow.log_metric("Valid best F1", best_metrics)
                mlflow.log_metric("Valid best loss", best_loss)
                for i in range(len(val_loss_history)):
                    mlflow.log_metric("Valid loss", val_loss_history[i])
                    mlflow.log_metric("Valid F1", val_metrics_history[i])

                mlflow.log_metric("Test F1 metric", test_metric)
                mlflow.log_metric("Throughput images/second", throughtput)
                mlflow.log_metric("Latency second", latency)
                for cls, f1 in per_class_metrics.items():
                    mlflow.log_metric(f"Test F1_class_{cls}", f1)
                mlflow.log_artifact(kwargs["config_path"])
                mlflow.pytorch.log_model(model, "model.pt")
                return None

        return wrapper

    return decorator
