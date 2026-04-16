"""
score_generator.py
------------------
Renders score_template.py into a concrete score.py file.
"""

import json
from pathlib import Path
from typing import Union

from src.azure.score_template import SCORE_PY_TEMPLATE
from src.model.preprocessor import FEATURE_COLUMNS


PathLike = Union[str, Path]


class ScoreScriptGenerator:

    def __init__(self, model_name: str, umbral_path: PathLike):
        self.model_name = model_name
        self.umbral_path = umbral_path

    def generate(self, output_path: PathLike) -> None:
        with open(self.umbral_path, "r") as f:
            umbral = json.load(f)["umbral"][0]

        script = SCORE_PY_TEMPLATE.format(
            model_name=self.model_name,
            umbral=umbral,
            feature_columns=FEATURE_COLUMNS,
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(script)

        print(f"[ScoreScriptGenerator] score.py generado en '{output_path}' (umbral={umbral:.6f}).")
