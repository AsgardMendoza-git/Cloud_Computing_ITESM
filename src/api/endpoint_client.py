"""
endpoint_client.py
------------------
Sends prediction requests to a deployed AzureML scoring endpoint.
"""

import json
from pathlib import Path
from typing import Union

import pandas as pd
import requests


PathLike = Union[str, Path]


class EndpointClient:

    def __init__(self, uri_path: PathLike):
        with open(uri_path, "r") as f:
            uris = json.load(f)["URI"]
        if not uris:
            raise RuntimeError(f"No scoring URI found in '{uri_path}'.")
        self.scoring_uri = uris[0]

    def predict(self, data: pd.DataFrame, timeout: int = 60) -> list:
        payload = {"data": [data.to_dict(orient="list")]}
        headers = {"Content-Type": "application/json"}

        response = requests.post(
            self.scoring_uri,
            headers=headers,
            data=json.dumps(payload),
            timeout=timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Request failed ({response.status_code}): {response.text}")

        body = response.json()
        while isinstance(body, str):
            body = json.loads(body)

        if isinstance(body, dict):
            if "error" in body:
                raise RuntimeError(f"Endpoint error: {body['error']}")
            if "predictions" in body:
                predictions = body["predictions"]
            else:
                raise RuntimeError(f"Unexpected response shape: {body}")
        elif isinstance(body, list):
            predictions = body
        else:
            raise RuntimeError(f"Unexpected response type: {type(body).__name__}")

        if len(predictions) != len(data):
            raise RuntimeError(
                f"Prediction count mismatch. Got {len(predictions)} for {len(data)} rows."
            )

        return predictions
