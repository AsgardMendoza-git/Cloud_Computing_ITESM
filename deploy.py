"""
deploy.py
---------
Clase AzureDeployer: encapsula el registro del modelo en AzureML y su
despliegue como web service en Azure Container Instances (ACI).

Basado en Deployer.ipynb del repositorio.

Dependencias: azureml-sdk
  pip install azureml-sdk
"""

from typing import List, Optional
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice


# --------------------------------------------------------------------------
# Script de scoring que se sube junto al modelo (se genera dinámicamente)
# --------------------------------------------------------------------------

SCORE_PY_TEMPLATE = '''
import json
import joblib
import numpy as np
import pandas as pd
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path("{model_name}")
    model = joblib.load(model_path)

def sigmoid(x):
    return [1 / (1 + np.exp(-v)) for v in x]

def run(raw_data):
    try:
        data = json.loads(raw_data)["data"][0]
        data = pd.DataFrame(data)

        # Misma lógica que zorrouno.processor.embbed para SalesLT.Product
        keep = ["StandardCost", "ListPrice", "Weight", "ProductCategoryID", "ProductModelID"]
        data = data[[c for c in keep if c in data.columns]].dropna()

        result = model.predict(data).tolist()
        result_sigmoid = sigmoid(result)
        umbral = {umbral}
        result_finals = [1 if x > umbral else 0 for x in result_sigmoid]

        return json.dumps(result_finals)
    except Exception as e:
        return json.dumps(str(e))
'''


class AzureDeployer:
    """Registra un modelo en AzureML y lo despliega como ACI web service."""

    def __init__(
        self,
        subscription_id: str,
        resource_group: str = "Papus",
        workspace_name: str = "workspace",
        location: str = "eastus",
        tenant_id: Optional[str] = None,
    ):
        """
        Parámetros
        ----------
        subscription_id : ID de la subscripción de Azure
        resource_group  : Nombre del resource group (se crea si no existe)
        workspace_name  : Nombre del workspace de AzureML
        location        : Región de Azure (ej. 'centralindia', 'eastus')
        tenant_id       : Tenant ID para cuentas EXATEC (opcional)
        """
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        self.location = location
        self.tenant_id = tenant_id

        self._ws = None  # type: Optional[Workspace]
        self._registered_model = None  # type: Optional[Model]
        self._service = None

    # ------------------------------------------------------------------
    # Workspace
    # ------------------------------------------------------------------

    def setup_workspace(self) -> None:
        """
        Crea o recupera el workspace de AzureML.
        Si se proporcionó tenant_id, usa InteractiveLoginAuthentication.
        """
        auth = None
        if self.tenant_id:
            auth = InteractiveLoginAuthentication(tenant_id=self.tenant_id)

        self._ws = Workspace.create(
            name=self.workspace_name,
            subscription_id=self.subscription_id,
            resource_group=self.resource_group,
            location=self.location,
            auth=auth,
            exist_ok=True,          # no falla si el workspace ya existe
        )
        print(f"[AzureDeployer] Workspace '{self.workspace_name}' listo.")

    # ------------------------------------------------------------------
    # Registro del modelo
    # ------------------------------------------------------------------

    def register_model(
        self,
        model_path: str = "model.pkl",
        model_name: str = "churn_model",
    ) -> None:
        """
        Sube y registra el modelo en el workspace de AzureML.

        Parámetros
        ----------
        model_path : Ruta local del archivo .pkl
        model_name : Nombre con el que quedará registrado en AzureML
        """
        if self._ws is None:
            raise RuntimeError("Debes llamar a setup_workspace() primero.")

        self._model_name = model_name
        self._registered_model = Model.register(
            model_path=model_path,
            model_name=model_name,
            workspace=self._ws,
        )
        print(f"[AzureDeployer] Modelo '{model_name}' registrado (v{self._registered_model.version}).")

    # ------------------------------------------------------------------
    # Generación de score.py
    # ------------------------------------------------------------------

    def generate_score_script(
        self,
        umbral_path: str = "umbral.json",
        output_path: str = "score.py",
    ) -> None:
        """
        Genera el script score.py que AzureML ejecuta en el contenedor.

        Parámetros
        ----------
        umbral_path : Ruta del JSON con el umbral calculado en model.py
        output_path : Dónde guardar el score.py generado
        """
        with open(umbral_path, "r") as f:
            umbral = json.load(f)["umbral"][0]

        script = SCORE_PY_TEMPLATE.format(
            model_name=self._model_name,
            umbral=umbral,
        )

        with open(output_path, "w") as f:
            f.write(script)

        print(f"[AzureDeployer] score.py generado en '{output_path}' (umbral={umbral:.6f}).")

    # ------------------------------------------------------------------
    # Despliegue
    # ------------------------------------------------------------------

    def deploy(
        self,
        service_name: str = "churn-service",
        score_script: str = "score.py",
        conda_packages: Optional[List[str]] = None,
        cpu_cores: float = 0.5,
        memory_gb: float = 1.0,
    ) -> str:
        """
        Despliega el modelo registrado como un ACI web service.

        Parámetros
        ----------
        service_name   : Nombre del servicio en AzureML
        score_script   : Ruta al script de scoring
        conda_packages : Paquetes conda para el entorno (default: pandas + scikit-learn)
        cpu_cores      : Núcleos de CPU asignados al contenedor
        memory_gb      : Memoria RAM asignada al contenedor

        Retorna
        -------
        scoring_uri : URL del endpoint para hacer predicciones
        """
        if self._ws is None or self._registered_model is None:
            raise RuntimeError("Llama a setup_workspace() y register_model() antes de deploy().")

        if conda_packages is None:
            conda_packages = ["pandas", "scikit-learn"]

        # Entorno virtual con las dependencias del modelo
        env = Environment("churn-env")
        env.python.conda_dependencies = CondaDependencies.create(
            conda_packages=conda_packages
        )

        inference_config = InferenceConfig(
            environment=env,
            entry_script=score_script,
        )
        aci_config = AciWebservice.deploy_configuration(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
        )

        service = Model.deploy(
            workspace=self._ws,
            name=service_name,
            models=[self._registered_model],
            inference_config=inference_config,
            deployment_config=aci_config,
            overwrite=True,
        )
        service.wait_for_deployment(show_output=True)
        self._service = service

        scoring_uri = service.scoring_uri
        print(f"[AzureDeployer] Servicio desplegado. URI: {scoring_uri}")

        # Guardar URI para que API.ipynb / api.py la consuma
        with open("uri.json", "w") as f:
            json.dump({"URI": [scoring_uri]}, f)
        print("[AzureDeployer] URI guardada en 'uri.json'.")

        return scoring_uri

    # ------------------------------------------------------------------
    # Pipeline completo (atajo)
    # ------------------------------------------------------------------

    def full_deploy(
        self,
        model_path: str = "model.pkl",
        umbral_path: str = "umbral.json",
        model_name: str = "churn_model",
        service_name: str = "churn-service",
    ) -> str:
        """
        Ejecuta workspace → registro → score.py → despliegue en un solo paso.

        Retorna
        -------
        scoring_uri del servicio desplegado
        """
        self.setup_workspace()
        self.register_model(model_path=model_path, model_name=model_name)
        self.generate_score_script(umbral_path=umbral_path)
        return self.deploy(service_name=service_name)
