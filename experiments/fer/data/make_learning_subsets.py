from pathlib import Path

import click
import splitfolders


@click.command()
@click.argument("dataset-dir", type=click.Path(exists=True))
@click.argument("output-dataset-dir", type=click.Path())
def make_learning_subsets(dataset_dir: Path | str, output_dataset_dir: Path | str):
    dataset_dir = Path(dataset_dir)
    Path(output_dataset_dir).mkdir(exist_ok=True, parents=True)



if __name__ == "__main__":
    make_learning_subsets()
