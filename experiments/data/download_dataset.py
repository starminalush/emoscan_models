from pathlib import Path

import click
from kaggle import api  # noqa: E402
from loguru import logger

api.authenticate()


@click.command()
@click.argument("external-dataset-path", type=click.Path(path_type=Path))
@click.argument("dataset-name", type=str)
def download_dataset(external_dataset_path: Path | str, dataset_name: str) -> None:
    """Download external dataset from kaggle by dataset name
    Args:
        dataset_name: kaggle dataset name
        external_dataset_path: downloaded dataset path
    """
    api.dataset_download_files(dataset_name, path=external_dataset_path, unzip=True)
    logger.info("finish download dataset")


if __name__ == "__main__":
    download_dataset()
