"""
main.py
-------
Pipeline completo: BD → Modelo → Despliegue en Azure.

Uso:
  python main.py

Las credenciales se leen de credentials.json (nunca hardcodeadas aquí).
Copia credentials.json.example → credentials.json y rellena tus datos.
"""

import json

# ============================================================
#  CARGAR CREDENCIALES (credentials.json está en .gitignore)
# ============================================================

with open("credentials.json", "r") as f:
    _creds = json.load(f)

# --- Azure SQL ---
SQL_SERVER   = _creds["sql_server"]
SQL_DATABASE = _creds["sql_database"]
SQL_USER     = _creds["sql_username"]
SQL_PASSWORD = _creds["sql_password"]
SQL_TABLE    = "SalesLT.Product"         # tabla con variable objetivo IsBlack (Color='Black')

# --- AzureML ---
AZURE_SUBSCRIPTION_ID = _creds["subscription_id"]
AZURE_RESOURCE_GROUP  = "Papus"
AZURE_WORKSPACE_NAME  = "workspace"
AZURE_LOCATION        = "eastus"

# --- Artefactos ---
MODEL_PATH  = "model.pkl"
UMBRAL_PATH = "umbral.json"
MODEL_NAME  = "churn_model"
SERVICE_NAME = "churn-service"


# ============================================================


from functions.database import DatabaseConnector
from functions.model    import ChurnModel
from functions.deploy   import AzureDeployer


def main():
    # ----------------------------------------------------------
    # PASO 1: Obtener datos de la base de datos SQL
    # ----------------------------------------------------------
    print("=" * 55)
    print("PASO 1 — Conectando a la base de datos Azure SQL...")
    print("=" * 55)

    with DatabaseConnector(SQL_SERVER, SQL_DATABASE, SQL_USER, SQL_PASSWORD) as db:
        data = db.get_churn_data(table=SQL_TABLE)

    print(data.columns)


    # ----------------------------------------------------------
    # PASO 2: Entrenar el modelo
    # ----------------------------------------------------------
    print("=" * 55)
    print("PASO 2 — Entrenando el modelo de churn...")
    print("=" * 55)

    churn_model = ChurnModel(test_size=0.2, random_state=42)
    churn_model.train(data)
    churn_model.save(model_path=MODEL_PATH, umbral_path=UMBRAL_PATH)
    print()

    # ----------------------------------------------------------
    # PASO 3: Desplegar en Azure
    # ----------------------------------------------------------
    print("=" * 55)
    print("PASO 3 — Desplegando en Azure Container Instances...")
    print("=" * 55)

    deployer = AzureDeployer(
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group=AZURE_RESOURCE_GROUP,
        workspace_name=AZURE_WORKSPACE_NAME,
        location=AZURE_LOCATION,
    )

    scoring_uri = deployer.full_deploy(
        model_path=MODEL_PATH,
        umbral_path=UMBRAL_PATH,
        model_name=MODEL_NAME,
        service_name=SERVICE_NAME,
    )

    # ----------------------------------------------------------
    # Resultado final
    # ----------------------------------------------------------
    print()
    print("=" * 55)
    print("✓ Pipeline completo.")
    print(f"  Endpoint para predicciones: {scoring_uri}")
    print("  Ahora puedes usar API.ipynb para consumir el servicio.")
    print("=" * 55)


if __name__ == "__main__":
    main()
