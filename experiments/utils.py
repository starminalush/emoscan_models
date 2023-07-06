import functools
import importlib
from typing import Any, Callable, Concatenate, ParamSpec, TypeAlias, TypeVar

import gdown
import mlflow
import yaml


Param = ParamSpec("Param")
RetType = TypeVar("RetType")
OriginalFunc: TypeAlias = Callable[Param, RetType]
DecoratedFunc: TypeAlias = Callable[Concatenate[tuple, Param], RetType]


def download_pretrained_model_from_gdrive(file_id: str, output_model_name: str):
    """Download model from gdrive based on file ID.

    Args:
        file_id: File ID in gdrive.
        output_model_name: Output file name.
    """
    uri = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(uri, output_model_name, quiet=False)


def load_config(config_path: str) -> dict[str, Any]:
    """Load config (for example, for train stage) from yaml file.

    Args:
        config_path: Input config path.

    Returns:
        Dict containing config params and values for each params.
    """
    with open(config_path) as src:
        config: dict = yaml.load(src, Loader=yaml.Loader)
    return config


def init_module(class_str: str) -> Callable:
    """Initialize class by import name.

    Args:
        class_str: Import path of class.

    Returns:
        The received attribute, which is supposed to be an object of the class.
    """
    module_path, class_name = class_str.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def mlflow_logger(experiment_name: str) -> Callable[[OriginalFunc], DecoratedFunc]:
    """Decorate train function for easy logging training result to mlflow server.

    Args:
        experiment_name: Custom experiment name of task.

    Returns:
        Result of decorated function.
    """

    def decorator(func: OriginalFunc) -> DecoratedFunc:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> RetType:
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
                for (val_loss, val_f1) in zip(val_loss_history, val_metrics_history):
                    mlflow.log_metric("Valid loss", val_loss)
                    mlflow.log_metric("Valid F1", val_f1)

                mlflow.log_metric("Test F1 metric", test_metric)
                mlflow.log_metric("Throughput images/second", throughtput)
                mlflow.log_metric("Latency second", latency)
                for class_label, f1 in per_class_metrics.items():
                    mlflow.log_metric(f"Test F1_class_{class_label}", f1)
                mlflow.log_artifact(kwargs["config_path"])
                mlflow.pytorch.log_model(model, "model.pt")
                return None

        return wrapper

    return decorator
