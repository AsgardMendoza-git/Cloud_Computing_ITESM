"""
preprocessor.py
---------------
Local replacement for zorrouno.processor.embbed.
Selects feature columns, derives the IsBlack target from Color, drops nulls.
"""

import pandas as pd


FEATURE_COLUMNS = [
    "StandardCost",
    "ListPrice",
    "Weight",
    "ProductCategoryID",
    "ProductModelID",
]

TARGET_COLUMN = "IsBlack"


def embbed(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    if "Color" in df.columns and TARGET_COLUMN not in df.columns:
        df[TARGET_COLUMN] = (df["Color"] == "Black").astype(int)

    keep = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c in df.columns]
    return df[keep].dropna()
