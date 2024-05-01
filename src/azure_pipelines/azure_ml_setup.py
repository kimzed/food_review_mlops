from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential
from azureml.core import Workspace
from settings import AZURE_ENV_FILE, AZURE_ENV_NAME, CONFIG_AZURE_ML, AzureMlConfig, AZURE_ML_CONFIG_FILE, COMPUTE_NAME
from azure.ai.ml.entities import Environment

def get_workspace() -> Workspace:
    """Get Azure ML workspace from config."""
    return Workspace.from_config(path=AZURE_ML_CONFIG_FILE)

def get_environment() -> Environment:
    env_docker_conda = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    conda_file=AZURE_ENV_FILE,
    name=AZURE_ENV_NAME,
    description="Environment for the food review mlops project.",
    )
    return env_docker_conda

def get_ml_client(config_info: AzureMlConfig) -> MLClient:
    """Instantiate MLClient using Azure credentials and workspace details."""
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=config_info.subscription_id,
        resource_group_name=config_info.resource_group,
        workspace_name=config_info.workspace_name,
    )


def get_compute_cluster(ml_client: MLClient, cluster_name: str) -> AmlCompute:
    """Ensure a compute cluster exists or create a new one."""
    try:
        cluster = ml_client.compute.get(cluster_name)
        print(f"Using existing cluster: {cluster_name}")
    except Exception:
        print("Creating a new CPU compute target...")
        cluster = AmlCompute(
            name=cluster_name,
            type="amlcompute",
            size="STANDARD_DS11_V2",
            min_instances=1,
            max_instances=1,
            idle_time_before_scale_down=120,
            tier="Dedicated",
        )
        cluster = ml_client.compute.begin_create_or_update(cluster).result()
        print(f"Created cluster: {cluster_name} with size {cluster.size}")
    return cluster


try:
    ML_CLIENT = get_ml_client(CONFIG_AZURE_ML)
    ML_WORKSPACE = get_workspace()
except:
    print("unable to connect to azure")

if __name__ == "__main__":
    compute_cluster = get_compute_cluster(ML_CLIENT, COMPUTE_NAME)
    env_docker_conda = get_environment()
    ML_CLIENT.environments.create_or_update(env_docker_conda)
