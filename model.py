"""
model.py
--------
Clase ChurnModel: encapsula el preprocesamiento, entrenamiento, evaluación
y persistencia del modelo de clasificación bancaria (variable objetivo: Exited).

El modelo base es una Regresión Lineal con umbral de sigmoide para producir
predicciones binarias — tal como se describe en Model.ipynb del repositorio.

Dependencias: scikit-learn, pandas, numpy, pickle
  pip install scikit-learn pandas numpy
"""

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from zorrouno import processor


class ChurnModel:
    """Entrena, evalúa y persiste un modelo de predicción de churn bancario."""

    # Columnas que NO son features (se descartan antes de entrenar)
    TARGET_COLUMN = "IsBlack"

    def __init__(self, test_size: float = 0.2, random_state: int = 42):  # type: ignore
        """
        Parámetros
        ----------
        test_size    : Proporción del conjunto de prueba (default 0.2 → 80/20)
        random_state : Semilla para reproducibilidad
        """
        self.test_size = test_size
        self.random_state = random_state

        self._model = None  # type: Optional[LinearRegression]
        self._umbral = None  # type: Optional[float]

    # ------------------------------------------------------------------
    # Propiedades de sólo lectura
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Preprocesamiento
    # ------------------------------------------------------------------

    @staticmethod
    def preprocess(data: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia el DataFrame aplicando la lógica de zorrouno.processor:
        elimina columnas no numéricas / irrelevantes.

        Parámetros
        ----------
        data : DataFrame crudo proveniente de la BD

        Retorna
        -------
        DataFrame listo para entrenar
        """
        return processor.embbed(data)

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x) -> list:
        return [1 / (1 + np.exp(-v)) for v in x]

    def train(self, data: pd.DataFrame) -> None:
        """
        Preprocesa los datos, hace train-test split y entrena la regresión lineal.
        Calcula también el umbral óptimo de sigmoide.

        Parámetros
        ----------
        data : DataFrame crudo con todas las columnas (incluida 'Exited')
        """
        # 1. Preprocesar
        clean_data = self.preprocess(data)

        # 2. Separar features / target
        X = clean_data.drop(self.TARGET_COLUMN, axis=1)
        y = clean_data[self.TARGET_COLUMN]

        # 3. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        # 4. Entrenar regresión lineal
        self._model = LinearRegression().fit(X_train, y_train)

        # 5. Calcular umbral: mínima probabilidad sigmoide entre las top-k=positivos predicciones
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

        # 6. Evaluación
        final_preds = [1 if y_pred_sigmoid[i] > self._umbral else 0 for i in range(len(y_pred_sigmoid))]
        print("[ChurnModel] Reporte de clasificación en test set:")
        print(classification_report(y_test, final_preds))
        print(f"[ChurnModel] Umbral de predicción calculado: {self._umbral:.6f}")

    # ------------------------------------------------------------------
    # Predicción
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> list:
        """
        Predice si cada cliente hará churn (1) o no (0).

        Parámetros
        ----------
        X : DataFrame con las mismas columnas usadas en el entrenamiento
            (sin la columna 'Exited')

        Retorna
        -------
        Lista de enteros: 1 = churn, 0 = no churn
        """
        X_clean = self.preprocess(X)
        raw = self.model.predict(X_clean)
        probs = self._sigmoid(raw)
        return [1 if p > self.umbral else 0 for p in probs]

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, model_path: str = "model.pkl", umbral_path: str = "umbral.json") -> None:
        """
        Guarda el modelo entrenado y el umbral en disco.

        Parámetros
        ----------
        model_path  : Ruta del archivo pickle del modelo
        umbral_path : Ruta del archivo JSON con el umbral
        """
        # Guardar modelo
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"[ChurnModel] Modelo guardado en '{model_path}'.")

        # Guardar umbral
        with open(umbral_path, "w") as f:
            json.dump({"umbral": [self.umbral]}, f)
        print(f"[ChurnModel] Umbral guardado en '{umbral_path}'.")

    @classmethod
    def load(cls, model_path: str = "model.pkl", umbral_path: str = "umbral.json") -> "ChurnModel":
        """
        Carga un modelo previamente guardado desde disco.

        Parámetros
        ----------
        model_path  : Ruta del archivo pickle del modelo
        umbral_path : Ruta del archivo JSON con el umbral

        Retorna
        -------
        Instancia de ChurnModel lista para predecir
        """
        instance = cls()

        with open(model_path, "rb") as f:
            instance._model = pickle.load(f)

        with open(umbral_path, "r") as f:
            instance._umbral = json.load(f)["umbral"][0]

        print(f"[ChurnModel] Modelo cargado desde '{model_path}' (umbral={instance._umbral:.6f}).")
        return instance
