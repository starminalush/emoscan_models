from pathlib import Path
from shutil import rmtree

import click
import splitfolders


@click.command()
@click.option("--dataset-dir", type=click.Path(exists=True, path_type=Path))
@click.option("--output-dataset-dir", type=click.Path(path_type=Path))
def make_learning_subsets(
    dataset_dir: Path | str, output_dataset_dir: Path | str
) -> None:
    """Split dataset on train. test and val subsets.

    Args:
        dataset_dir: Path of dataset.
        output_dataset_dir: Path of divided dataset.
    """
    output_dataset_dir.mkdir(exist_ok=True, parents=True)
    splitfolders.ratio(
        dataset_dir, output=output_dataset_dir, seed=42, ratio=(0.8, 0.1, 0.1),  # noqa: WPS432
    )
    for subset in {"train", "val", "test"}:
        (output_dataset_dir / subset / "angry").mkdir(exist_ok=True)
        (output_dataset_dir / subset / "anger").rename(
            output_dataset_dir / subset / "angry"
        )
        rmtree((output_dataset_dir / subset / "contempt"))


if __name__ == "__main__":
    make_learning_subsets()
