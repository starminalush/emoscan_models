from pathlib import Path

import click
from kaggle import api  # noqa: E402
from loguru import logger

api.authenticate()


@click.command()
@click.option("--external-dataset-dir", type=click.Path(path_type=Path))
@click.option("--dataset-name", type=str)
def download_dataset(external_dataset_dir: Path | str, dataset_name: str) -> None:
    """Download external dataset from kaggle by dataset name.

    Args:
        dataset_name: Kaggle dataset name.
        external_dataset_dir: Downloaded dataset path.
    """
    api.dataset_download_files(dataset_name, path=external_dataset_dir, unzip=True)
    logger.info("finish download dataset")


if __name__ == "__main__":
    download_dataset()
