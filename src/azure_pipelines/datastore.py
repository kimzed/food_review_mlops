from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Datastore
from settings import ML_CLIENT


# Register the datastore
blob_datastore = Datastore(
    name="aa_capstone_amazon_review_data",
    description="Datastore for the amazon review project",
    account_name="aa_capstone_storage",
    container_name="aa_capstone_container",
    credentials={"sas_token": "your_sas_token_or_storage_account_key"},
    datastore_type="AzureBlob"
)

ML_CLIENT.datastores.create_or_update(blob_datastore)