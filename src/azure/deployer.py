"""
deployer.py
-----------
Orchestrates the full Azure deployment pipeline:
workspace -> model registration -> score script -> service deployment.
"""

from pathlib import Path
from typing import Optional, Union

from src.azure.workspace import WorkspaceManager
from src.azure.model_registrar import ModelRegistrar
from src.azure.score_generator import ScoreScriptGenerator
from src.azure.service_deployer import ServiceDeployer


PathLike = Union[str, Path]


class AzureDeployer:

    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        workspace_name: str,
        location: str,
        tenant_id: Optional[str] = None,
    ):
        self.workspace_manager = WorkspaceManager(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            location=location,
            tenant_id=tenant_id,
        )

    def full_deploy(
        self,
        model_path: PathLike,
        umbral_path: PathLike,
        score_path: PathLike,
        uri_path: PathLike,
        model_name: str,
        service_name: str,
    ) -> str:
        ws = self.workspace_manager.get_or_create()

        registered = ModelRegistrar(ws).register(model_path=model_path, model_name=model_name)

        ScoreScriptGenerator(model_name=model_name, umbral_path=umbral_path).generate(
            output_path=score_path
        )

        return ServiceDeployer(workspace=ws, registered_model=registered).deploy(
            service_name=service_name,
            score_script=score_path,
            uri_output_path=uri_path,
        )
