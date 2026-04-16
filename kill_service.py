"""
kill_service.py
---------------
Deletes the deployed ACI web service to stop runtime charges.

Usage:
    python kill_service.py
"""

from src.config import (
    load_credentials,
    AZURE_RESOURCE_GROUP,
    AZURE_WORKSPACE_NAME,
    SERVICE_NAME,
    URI_PATH,
)
from src.api import ServiceManager


CONFIRM_DELETE = True


def main() -> None:
    if not CONFIRM_DELETE:
        raise RuntimeError(
            "Borrado bloqueado. Define CONFIRM_DELETE=True para eliminar el servicio."
        )

    creds = load_credentials()

    manager = ServiceManager(
        subscription_id=creds["subscription_id"],
        resource_group=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_WORKSPACE_NAME,
    )
    manager.delete(SERVICE_NAME)
    ServiceManager.clear_uri_file(URI_PATH)


if __name__ == "__main__":
    main()
