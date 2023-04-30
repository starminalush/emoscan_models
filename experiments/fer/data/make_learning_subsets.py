from pathlib import Path

import click
import splitfolders


@click.command()
@click.argument("dataset-dir", type=click.Path(exists=True))
@click.argument("output-dataset-dir", type=click.Path())
def make_learning_subsets(dataset_dir: Path | str, output_dataset_dir: Path | str):
    dataset_dir = Path(dataset_dir)
    Path(output_dataset_dir).mkdir(exist_ok=True, parents=True)
    train_dataset_len = len([f for f in (dataset_dir / "train").rglob("*.jpg")])
    test_dataset_len = len([f for f in (dataset_dir / "test").rglob("*.jpg")])
    valid_dataset_ratio = test_dataset_len / train_dataset_len
    splitfolders.ratio(
        (dataset_dir / "train"),
        output=output_dataset_dir,
        seed=42,
        ratio=(1.0 - valid_dataset_ratio, valid_dataset_ratio),
    )


if __name__ == "__main__":
    make_learning_subsets()
