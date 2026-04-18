"""
churn_model.py
--------------
ChurnModel: trains a linear regression with sigmoid threshold for binary
classification of products (target: IsBlack).
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.model.preprocessor import embbed, TARGET_COLUMN


class ChurnModel:

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

        self._model: Optional[LinearRegression] = None
        self._umbral: Optional[float] = None

    @property
    def model(self) -> LinearRegression:
        if self._model is None:
            raise RuntimeError("El modelo aún no ha sido entrenado. Llama a train().")
        return self._model

    @property
    def umbral(self) -> float:
        if self._umbral is None:
            raise RuntimeError("El umbral aún no ha sido calculado. Llama a train().")
        return self._umbral

    @staticmethod
    def _sigmoid(x) -> list:
        return [1 / (1 + np.exp(-v)) for v in x]

    def train(self, data: pd.DataFrame) -> None:
        clean_data = embbed(data)

        X = clean_data.drop(TARGET_COLUMN, axis=1)
        y = clean_data[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        self._model = LinearRegression().fit(X_train, y_train)

        y_pred_raw = self._model.predict(X_test)
        y_pred_sigmoid = self._sigmoid(y_pred_raw)

        sorted_indices = sorted(
            range(len(y_pred_sigmoid)),
            key=lambda i: y_pred_sigmoid[i],
            reverse=True,
        )
        how_many_ones = int(y_test.value_counts().get(1, 1))
        top_indices = sorted_indices[:how_many_ones]
        self._umbral = min(y_pred_sigmoid[i] for i in top_indices)

        final_preds = [1 if y_pred_sigmoid[i] > self._umbral else 0 for i in range(len(y_pred_sigmoid))]
        print("[ChurnModel] Reporte de clasificación en test set:")
        print(classification_report(y_test, final_preds))
        print(f"[ChurnModel] Umbral calculado: {self._umbral:.6f}")

    def predict(self, X: pd.DataFrame) -> list:
        X_clean = embbed(X)
        raw = self.model.predict(X_clean)
        probs = self._sigmoid(raw)
        return [1 if p > self.umbral else 0 for p in probs]

    def set_state(self, model: LinearRegression, umbral: float) -> None:
        self._model = model
        self._umbral = umbral
