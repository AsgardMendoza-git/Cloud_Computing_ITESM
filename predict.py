"""
predict.py
----------
Loads prediction_instances.csv, calls the deployed endpoint, prints results.

Usage:
    python predict.py
"""

from src.config import PREDICTION_CSV, URI_PATH
from src.api import InputLoader, EndpointClient


def main() -> None:
    data = InputLoader(PREDICTION_CSV).load()
    print(f"[predict] {len(data)} filas cargadas desde '{PREDICTION_CSV}'.")

    client = EndpointClient(URI_PATH)
    predictions = client.predict(data)

    result = data.copy()
    result["Prediction"] = predictions
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
