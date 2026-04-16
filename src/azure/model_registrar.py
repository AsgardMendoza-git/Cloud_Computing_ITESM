"""
model_registrar.py
------------------
Registers a local .pkl model in an AzureML workspace.
"""

from pathlib import Path
from typing import Union

from azureml.core import Workspace
from azureml.core.model import Model


PathLike = Union[str, Path]


class ModelRegistrar:

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def register(self, model_path: PathLike, model_name: str) -> Model:
        registered = Model.register(
            model_path=str(model_path),
            model_name=model_name,
            workspace=self.workspace,
        )
        print(f"[ModelRegistrar] Modelo '{model_name}' registrado (v{registered.version}).")
        return registered
