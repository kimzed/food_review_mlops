import uuid

from azure_pipelines.azure_ml_setup import ML_CLIENT
from settings import MODEL_NAME
from azure.ai.ml.entities import ManagedOnlineDeployment

def main():
    # Create a unique name for the endpoint
    online_endpoint_name = "sentiment-analysis-endpoint-" + str(uuid.uuid4())[:8]

    latest_model_version = max(
        [int(m.version) for m in ML_CLIENT.models.list(name=MODEL_NAME)]
    )

    print(latest_model_version)

    # Choose the latest version of our registered model for deployment
    model = ML_CLIENT.models.get(name=MODEL_NAME, version=latest_model_version)
    

    # define an online deployment
    test_deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=online_endpoint_name,
        model=model,
        instance_type="Standard_D2as_v4",
        instance_count=1,
    )
    test_deployment = ML_CLIENT.online_deployments.begin_create_or_update(
    test_deployment).result()

if __name__ == "__main__":
    main()