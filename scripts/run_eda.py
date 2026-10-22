import argparse
import json
import os
from pathlib import Path

import mlflow
import nbformat
from mlflow import MlflowClient
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mlflow/eda"))
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["MLFLOW_TRACKING_URI"] = arguments.tracking_uri
    os.environ["GENUI_EDA_OUTPUT_DIR"] = str(output)
    notebook = nbformat.read(root / "notebooks/eda.ipynb", as_version=4)
    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    )
    client.execute()
    executed = output / "eda.ipynb"
    nbformat.write(notebook, executed)
    result = json.loads((output / "run.json").read_text())
    mlflow.set_tracking_uri(arguments.tracking_uri)
    MlflowClient().log_artifact(result["run_id"], str(executed), artifact_path="eda")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
