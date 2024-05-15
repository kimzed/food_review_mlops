import uuid

from src.azure_pipelines.azure_ml_setup import ML_CLIENT
from settings import AZURE_ENV_NAME, MODEL_NAME
from azure.ai.ml.entities import ManagedOnlineDeployment, ManagedOnlineEndpoint, CodeConfiguration,Environment

def main():
    # Create a unique name for the endpoint
    online_endpoint_name = "sentiment-analysis-endpoint" # + str(uuid.uuid4())[:8]

    latest_model_version = max(
        [int(m.version) for m in ML_CLIENT.models.list(name=MODEL_NAME)]
    )
    env = Environment(
    conda_file="./environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )


    # create an online endpoint
    endpoint = ManagedOnlineEndpoint(
        name=online_endpoint_name,
        description="endpoint for the sentiment analysis model",
        auth_mode="key",
    )

    ML_CLIENT.online_endpoints.begin_create_or_update(endpoint)

    # define an online deployment
    deployment = ManagedOnlineDeployment(
        name="sentiment-analyzer-deployment",
        endpoint_name=online_endpoint_name,
        environment=f"{AZURE_ENV_NAME}@latest",
        code_configuration=CodeConfiguration(
        code="src/azure_pipelines/", scoring_script="score.py"
    ),
        instance_type="Standard_DS3_v2",
        instance_count=1,
    )

    ML_CLIENT.online_deployments.begin_create_or_update(
    deployment=deployment)

if __name__ == "__main__":
    main()