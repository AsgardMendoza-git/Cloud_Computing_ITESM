"""
service_manager.py
------------------
Deletes a deployed AzureML web service to stop ACI runtime charges.
"""

import json
from pathlib import Path
from typing import Union

from azureml.core import Workspace
from azureml.core.webservice import Webservice


PathLike = Union[str, Path]


class ServiceManager:

    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        workspace_name: str,
    ):
        self.workspace = Workspace.get(
            name=workspace_name,
            subscription_id=subscription_id,
            resource_group=resource_group,
        )

    def delete(self, service_name: str) -> None:
        service = Webservice(name=service_name, workspace=self.workspace)
        print(f"[ServiceManager] Borrando servicio: {service_name}")
        service.delete()
        print("[ServiceManager] Servicio borrado.")

    @staticmethod
    def clear_uri_file(uri_path: PathLike) -> None:
        try:
            with open(uri_path, "w") as f:
                json.dump({"URI": []}, f)
            print(f"[ServiceManager] '{uri_path}' limpiado.")
        except Exception as e:
            print(f"[ServiceManager] No se pudo limpiar '{uri_path}': {e}")
