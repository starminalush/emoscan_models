from pathlib import Path

import click
import mlflow
import onnx
import torch
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
        run_id: ID of mlflow run.
        artifact_model_uri: Model uri path.
        onnx_model_path: Output onnx model path.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = mlflow.pytorch.load_model(artifact_model_uri)
    model.to(device)
    model.eval()
    input_dummy_tensor: torch.Tensor = torch.randn(1, 3, 224, 224, requires_grad=True).to(device)
    torch.onnx.export(
        model,
        input_dummy_tensor,
        onnx_model_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["cls", "feature", "heads"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(onnx_model_path)


if __name__ == "__main__":
    cnvt_torch_to_onnx()
