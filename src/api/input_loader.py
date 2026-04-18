"""
input_loader.py
---------------
Loads a CSV of prediction instances and validates required feature columns.
"""

from pathlib import Path
from typing import Union

import pandas as pd

from src.model.preprocessor import FEATURE_COLUMNS


PathLike = Union[str, Path]


class InputLoader:

    def __init__(self, csv_path: PathLike):
        self.csv_path = csv_path

    def load(self) -> pd.DataFrame:
        data = pd.read_csv(self.csv_path)

        missing = [c for c in FEATURE_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns in input CSV: {missing}")

        return data[FEATURE_COLUMNS].copy()
