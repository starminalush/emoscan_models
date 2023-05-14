import click
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
from kaggle import api  # noqa: E402

api.authenticate()


@click.command()
@click.argument("external-dataset-path", type=click.Path())
def download_dataset(external_dataset_path: str):
    """Download external dataset from kaggle. Link: https://www.kaggle.com/datasets/msambare/fer2013.
    Args:
        external_dataset_path: download dataset path
    """
    ds_name: str = "msambare/fer2013"
    api.dataset_download_files(ds_name, path=external_dataset_path, unzip=True)
    logger.info("finish download dataset")


if __name__ == "__main__":
    download_dataset()
