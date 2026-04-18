"""
persistence.py
--------------
Save and load the trained model and threshold to/from disk.
"""

import json
import pickle
from pathlib import Path
from typing import Union

from src.model.churn_model import ChurnModel


PathLike = Union[str, Path]


class ModelStorage:

    @staticmethod
    def save(model: ChurnModel, model_path: PathLike, umbral_path: PathLike) -> None:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(umbral_path).parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, "wb") as f:
            pickle.dump(model.model, f)
        print(f"[ModelStorage] Modelo guardado en '{model_path}'.")

        with open(umbral_path, "w") as f:
            json.dump({"umbral": [model.umbral]}, f)
        print(f"[ModelStorage] Umbral guardado en '{umbral_path}'.")

    @staticmethod
    def load(model_path: PathLike, umbral_path: PathLike) -> ChurnModel:
        with open(model_path, "rb") as f:
            sk_model = pickle.load(f)

        with open(umbral_path, "r") as f:
            umbral = json.load(f)["umbral"][0]

        instance = ChurnModel()
        instance.set_state(sk_model, umbral)
        print(f"[ModelStorage] Modelo cargado desde '{model_path}' (umbral={umbral:.6f}).")
        return instance
