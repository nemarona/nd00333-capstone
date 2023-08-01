import json
import numpy as np
from pathlib import Path
import joblib
import os


def init():
    global model

    # best_run_id = "HD_cc76c1bb-a826-450b-aa99-e566e43ad2ed_41"

    output_path = Path(os.getenv('AZUREML_MODEL_DIR')) / "outputs"
    assert output_path.exists(), f"Path not found: {output_path.absolute()}"

    model_paths = list(output_path.glob("model_*.joblib"))
    model_path = model_paths[0]
    assert model_path.exists(), f"Path not found: {model_path.absolute()}"

    model = joblib.load(model_path)


def run(raw_data):
    data = np.array(json.loads(raw_data)["data"])
    # make prediction
    y_hat = model.predict(data)
    # you can return any data type as long as it is JSON-serializable
    return y_hat.tolist()
