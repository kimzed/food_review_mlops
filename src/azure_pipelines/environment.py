from pathlib import WindowsPath
from azure.ai.ml.entities import Environment
from azureml.core import Environment
from src.azure_pipelines.azure_ml_setup import ML_CLIENT, get_ml_client, get_workspace
from settings import AZURE_ENV_FILE, AZURE_ENV_NAME, AZURE_ML_CONFIG_FILE, CONFIG_AZURE_ML
from azureml.core import Workspace

# custom_job_env = Environment(
#     name=AZURE_ENV_NAME,
#     description="Custom environment for ml project on amazon review dataset",
#     conda_file=AZURE_ENV_FILE,
#     # docker image
#     #image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest",
# )

# custom_job_env = ML_CLIENT.environments.create_or_update(custom_job_env)

# food_review_env = Environment.from_conda_specification(
#     name=AZURE_ENV_NAME,
#     file_path=AZURE_ENV_FILE,
# )

# food_review_env.register(workspace=ML_WORKSPACE)

# from azureml.core.model import InferenceConfig
# from azureml.core import Environment
# from azureml.core.conda_dependencies import CondaDependencies

# Workspace.from_config(path=AZURE_ML_CONFIG_FILE)
# environment = Environment('my-sklearn-environment')
# environment.python.conda_dependencies = CondaDependencies.create(pip_packages=[
#     'azureml-defaults',
#     'inference-schema[numpy-support]',
#     'joblib',
#     'numpy',
#     'pandas',
# ])


# env_docker_image = Environment(
#     image="pytorch/pytorch:latest",
#     name="docker-image-example",
#     description="Environment created from a Docker image.",
# )
# ml_client.environments.create_or_update(env_docker_image)
# 2.2 Create a custom environment from Docker build context configuration
# In this sample we will use a local docker build configuration to create an environment
from azure.ai.ml.entities import Environment, BuildContext
path_dockerfile = WindowsPath(r"C:\Users\cedric.baron\source\repos\Fine-Food-Review-Analysis\config\Dockerfile")
docker_context_directory = WindowsPath(r"C:\Users\cedric.baron\source\repos\Fine-Food-Review-Analysis\config")
ML_CLIENT = get_ml_client(CONFIG_AZURE_ML)
env_docker_context = Environment(
    build=BuildContext(dockerfile_path=path_dockerfile, path=docker_context_directory),
    name="docker-context-example",
    description="Environment created from a Docker context.",
)
ML_CLIENT.environments.create_or_update(env_docker_context)

print(f"code completed, path exists= {path_dockerfile.exists()}")
