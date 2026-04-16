"""
service_deployer.py
-------------------
Deploys a registered model as an Azure Container Instances (ACI) web service.
"""

import json
from pathlib import Path
from typing import List, Optional, Union

from azureml.core import Workspace
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice


PathLike = Union[str, Path]


class ServiceDeployer:

    def __init__(self, workspace: Workspace, registered_model: Model):
        self.workspace = workspace
        self.registered_model = registered_model

    def deploy(
        self,
        service_name: str,
        score_script: PathLike,
        uri_output_path: PathLike,
        conda_packages: Optional[List[str]] = None,
        pip_packages: Optional[List[str]] = None,
        cpu_cores: float = 0.5,
        memory_gb: float = 1.0,
    ) -> str:
        if conda_packages is None:
            conda_packages = ["pandas", "scikit-learn", "numpy"]
        if pip_packages is None:
            pip_packages = ["azureml-defaults", "joblib"]

        env = Environment("churn-env")
        env.python.conda_dependencies = CondaDependencies.create(
            conda_packages=conda_packages,
            pip_packages=pip_packages,
        )

        inference_config = InferenceConfig(
            environment=env,
            entry_script=str(score_script),
        )
        aci_config = AciWebservice.deploy_configuration(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
        )

        service = Model.deploy(
            workspace=self.workspace,
            name=service_name,
            models=[self.registered_model],
            inference_config=inference_config,
            deployment_config=aci_config,
            overwrite=True,
        )
        service.wait_for_deployment(show_output=True)

        scoring_uri = service.scoring_uri
        print(f"[ServiceDeployer] Servicio desplegado. URI: {scoring_uri}")

        Path(uri_output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(uri_output_path, "w") as f:
            json.dump({"URI": [scoring_uri]}, f)
        print(f"[ServiceDeployer] URI guardada en '{uri_output_path}'.")

        return scoring_uri
