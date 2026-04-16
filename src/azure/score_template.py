"""
score_template.py
-----------------
Template for the score.py script that runs inside the AzureML container.
"""

SCORE_PY_TEMPLATE = '''
import json
import joblib
import numpy as np
import pandas as pd
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path("{model_name}")
    model = joblib.load(model_path)

def sigmoid(x):
    return [1 / (1 + np.exp(-v)) for v in x]


def _payload_to_dataframe(raw_data):
    payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data

    if isinstance(payload, dict):
        if "data" in payload:
            data_payload = payload["data"]
        elif "inputs" in payload:
            data_payload = payload["inputs"]
        else:
            raise KeyError("Expected payload key 'data' or 'inputs'.")
    elif isinstance(payload, list):
        data_payload = payload
    else:
        raise TypeError("Payload must be a dict, list, or JSON string.")

    if isinstance(data_payload, dict):
        return pd.DataFrame(data_payload)

    if isinstance(data_payload, list):
        if len(data_payload) == 0:
            return pd.DataFrame()

        if len(data_payload) == 1 and isinstance(data_payload[0], dict):
            first = data_payload[0]
            if all(isinstance(v, list) for v in first.values()):
                return pd.DataFrame(first)

        return pd.DataFrame(data_payload)

    raise TypeError("'data'/'inputs' value must be dict or list.")

def run(raw_data):
    try:
        data = _payload_to_dataframe(raw_data)

        if data.empty:
            return {{"error": "Input payload produced an empty DataFrame."}}

        keep = {feature_columns}
        missing = [c for c in keep if c not in data.columns]
        if missing:
            return {{
                "error": "Missing required feature columns.",
                "required_columns": keep,
                "missing_columns": missing,
            }}

        data = data[keep].dropna()
        if data.empty:
            return {{"error": "No rows left after dropping null feature values."}}

        result = model.predict(data).tolist()
        result_sigmoid = sigmoid(result)
        umbral = {umbral}
        result_finals = [1 if x > umbral else 0 for x in result_sigmoid]

        return {{"predictions": result_finals}}
    except Exception as e:
        return {{"error": str(e)}}
'''
