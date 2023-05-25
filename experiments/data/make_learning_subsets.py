from pathlib import Path
from shutil import rmtree

import click
import splitfolders



@click.command()
@click.argument("dataset-dir", type=click.Path(exists=True))
@click.argument("output-dataset-dir", type=click.Path())
def make_learning_subsets(dataset_dir: Path | str, output_dataset_dir: Path | str):
    dataset_dir = Path(dataset_dir)
    output_dataset_dir = Path(output_dataset_dir)
    Path(output_dataset_dir).mkdir(exist_ok=True, parents=True)
    splitfolders.ratio(
        dataset_dir, output=output_dataset_dir, seed=42, ratio=(0.8, 0.1, 0.1)
    )
    for subset in ["train", "val", "test"]:
        (output_dataset_dir / subset / "angry").mkdir(exist_ok=True)
        (output_dataset_dir / subset / "anger").rename(
            output_dataset_dir / subset / "angry"
        )
        rmtree((output_dataset_dir / subset / "contempt"))


if __name__ == "__main__":
    make_learning_subsets()
