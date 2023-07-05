from pathlib import Path

import click
import onnx
import torch
import mlflow
from dotenv import load_dotenv

load_dotenv()


@click.command()
@click.option("--run-id", type=str)
@click.option("--artifact-model-uri", type=str)
@click.option(
    "--onnx_model-path",
    type=click.Path(path_type=Path),
    required=False,
    default="onnx_model.onnx",
)
def cnvt_torch_to_onnx(run_id: str, artifact_model_uri: str, onnx_model_path: str):
    """Get model from mlflow, convert to onnx and write to run.
    Args:
        run_id: id of mlflow run
        artifact_model_uri: model uri path
        onnx_model_path: output onnx model path

    Returns:

    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # загружаем модель c mlflow
    model = mlflow.pytorch.load_model(artifact_model_uri)
    model.to(device)
    batch_size = 1  # just a random number
    model.eval()
    print(model)
    x = torch.randn(1, 3, 224, 224, requires_grad=True).to(device)
    r = model(x)
    _, predicted = torch.max(r[0], 1)
    print(predicted)
    print(onnx_model_path)

    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        onnx_model_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["cls", "feature", "heads"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )

    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(onnx_model_path)


if __name__ == "__main__":
    cnvt_torch_to_onnx()