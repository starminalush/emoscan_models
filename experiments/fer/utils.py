import functools
from time import time

import yaml
from loguru import logger


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
