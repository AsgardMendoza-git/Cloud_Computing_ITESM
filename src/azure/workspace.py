"""
workspace.py
------------
Creates or retrieves an AzureML workspace.
"""

from typing import Optional

from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication


class WorkspaceManager:

    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        workspace_name: str,
        location: str,
        tenant_id: Optional[str] = None,
    ):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        self.location = location
        self.tenant_id = tenant_id

    def get_or_create(self) -> Workspace:
        auth = InteractiveLoginAuthentication(tenant_id=self.tenant_id) if self.tenant_id else None

        ws = Workspace.create(
            name=self.workspace_name,
            subscription_id=self.subscription_id,
            resource_group=self.resource_group,
            location=self.location,
            auth=auth,
            exist_ok=True,
        )
        print(f"[WorkspaceManager] Workspace '{self.workspace_name}' listo.")
        return ws
