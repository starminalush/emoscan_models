import click
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
from kaggle import api  # noqa: E402

api.authenticate()


@click.command()
@click.argument("external-dataset-path", type=click.Path())
@click.argument("dataset-name", type=str)
def download_dataset(external_dataset_path: str, dataset_name: str):
    """Download external dataset from kaggle by dataset name
    Args:
        dataset_name: kaggle dataset name
        external_dataset_path: download dataset path
    """
    api.dataset_download_files(dataset_name, path=external_dataset_path, unzip=True)
    logger.info("finish download dataset")


if __name__ == "__main__":
    download_dataset()
