import json
import pandas as pd
from pathlib import Path
import joblib
import os


def init():
    global model, scaler

    output_path = Path(os.getenv("AZUREML_MODEL_DIR")) / "outputs"
    assert output_path.exists(), f"Path not found: {output_path.absolute()}"

    model_path = output_path / "model.joblib"
    scaler_path = output_path / "scaler.joblib"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)


def run(raw_data):
    # Preprocess
    scalable_features = [
        "age",
        "creatinine_phosphokinase",
        "ejection_fraction",
        "platelets",
        "serum_creatinine",
        "serum_sodium",
        "time",
    ]
    data = pd.DataFrame(json.loads(raw_data)["data"])
    data[scalable_features] = scaler.transform(data[scalable_features])
    # Make prediction
    predictions = model.predict(data)
    # You can return any data type as long as it is JSON-serializable
    return predictions.tolist()
