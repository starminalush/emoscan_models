import functools
import importlib
from time import time

import gdown
import yaml
from loguru import logger

import mlflow


def download_pretrained_model_from_gdrive(file_id: str, output_model_name: str):
    uri = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(uri, output_model_name, quiet=False)


def load_config(config_path: str):
    with open(config_path) as src:
        config: dict = yaml.load(src, Loader=yaml.Loader)
    return config


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        time_elapsed: float = time() - start_time
        logger.info(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        return result

    return wrapper


def init_module(class_str: str):
    module_path, class_name = class_str.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def mlflow_logger(experiment_name):
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
                    val_loss_history_cls,
                    val_loss_history_af,
                    val_loss_histoyp_pt,
                ) = func(*args, **kwargs)
                mlflow.log_metric("Valid best F1", best_metrics)
                mlflow.log_metric("Valid best loss", best_loss)
                for i in range(len(val_loss_history)):
                    mlflow.log_metric("Valid loss", val_loss_history[i])
                    mlflow.log_metric("Valid F1", val_metrics_history[i])
                    mlflow.log_metric("Valid cls loss", val_loss_history_cls[i])
                    mlflow.log_metric("Valid af loss", val_loss_history_af[i])
                    mlflow.log_metric("Valid pt loss", val_loss_histoyp_pt[i])

                mlflow.log_metric("Test F1 metric", test_metric)
                mlflow.log_metric("Throughtput images/second", throughtput)
                mlflow.log_metric("Latency second", latency)
                for cls, f1 in per_class_metrics.items():
                    mlflow.log_metric(f"Test F1_class_{cls}", f1)
                mlflow.log_artifact(kwargs["config_path"])
                mlflow.pytorch.log_model(model, "model.pt")
                return None

        return wrapper

    return decorator
