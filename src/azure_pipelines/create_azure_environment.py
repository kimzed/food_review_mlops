from azure.ai.ml.entities import Environment
from src.azure_pipelines.azure_ml_setup import ML_CLIENT
from settings import AZURE_ENV_FILE, AZURE_ENV_NAME, AZURE_ENV_FILE


env_docker_conda = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    conda_file=AZURE_ENV_FILE,
    name=AZURE_ENV_NAME,
    description="Environment for the food review mlops project.",
)
ML_CLIENT.environments.create_or_update(env_docker_conda)
