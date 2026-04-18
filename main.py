"""
main.py
-------
Pipeline: Azure SQL -> train ChurnModel -> deploy as ACI web service.

Usage:
    python main.py
"""

from src.config import (
    load_credentials,
    ensure_artifacts_dir,
    SQL_TABLE,
    AZURE_RESOURCE_GROUP,
    AZURE_WORKSPACE_NAME,
    AZURE_LOCATION,
    MODEL_NAME,
    SERVICE_NAME,
    MODEL_PATH,
    UMBRAL_PATH,
    SCORE_PATH,
    URI_PATH,
)
from src.database import DatabaseConnector
from src.model import ChurnModel, ModelStorage
from src.azure import AzureDeployer


def main() -> None:
    creds = load_credentials()
    ensure_artifacts_dir()

    print("=" * 55)
    print("PASO 1 — Conectando a la base de datos Azure SQL...")
    print("=" * 55)
    with DatabaseConnector(
        server=creds["sql_server"],
        database=creds["sql_database"],
        username=creds["sql_username"],
        password=creds["sql_password"],
    ) as db:
        data = db.get_churn_data(table=SQL_TABLE)

    print("=" * 55)
    print("PASO 2 — Entrenando el modelo de churn...")
    print("=" * 55)
    model = ChurnModel(test_size=0.2, random_state=42)
    model.train(data)
    ModelStorage.save(model, model_path=MODEL_PATH, umbral_path=UMBRAL_PATH)

    print("=" * 55)
    print("PASO 3 — Desplegando en Azure Container Instances...")
    print("=" * 55)
    deployer = AzureDeployer(
        subscription_id=creds["subscription_id"],
        resource_group=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_WORKSPACE_NAME,
        location=AZURE_LOCATION,
    )
    scoring_uri = deployer.full_deploy(
        model_path=MODEL_PATH,
        umbral_path=UMBRAL_PATH,
        score_path=SCORE_PATH,
        uri_path=URI_PATH,
        model_name=MODEL_NAME,
        service_name=SERVICE_NAME,
    )

    print()
    print("=" * 55)
    print("Pipeline completo.")
    print(f"Endpoint: {scoring_uri}")
    print("Ahora puedes ejecutar `python predict.py` para consumirlo.")
    print("=" * 55)


if __name__ == "__main__":
    main()
