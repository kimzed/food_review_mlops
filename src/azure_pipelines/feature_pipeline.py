from src.azure_pipelines.azure_ml_setup import ML_CLIENT
from settings import NAME_DATA_ASSET_RAW, AZURE_ENV_NAME, COMPUTE_NAME
from azure.ai.ml import Input
from azure.ai.ml import dsl, Input, Output
from azure.ai.ml import command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.storage.blob import BlobServiceClient

def preprocessing_pipeline(data: Input, output: dict[str:Output], out_path):
    return command(
        inputs=dict(
            data=Input(
            type="uri_file",
            path="azureml://subscriptions/f3a81a7f-b385-4cf3-aade-5da1a4dcb516/resourcegroups/aa_capstone/workspaces/aa_capstone/datastores/workspaceblobstore/paths/LocalUpload/7a0946a98fbc7c6e52d3cb8a93b78c92/Reviews.csvtest_subset.csv",
        ),outputs=out_path,),
        outputs=output,
        code="./",
        command="python src/preprocessing.py --data ${{inputs.data}} --output ${{inputs.outputs}}",
        environment=f"{AZURE_ENV_NAME}@latest",
        compute=COMPUTE_NAME,
        display_name="preprocessing_review_data",
    )

output_path = "azureml://subscriptions/f3a81a7f-b385-4cf3-aade-5da1a4dcb516/resourcegroups/aa_capstone/workspaces/aa_capstone/datastores/preprocessed_data/paths/LocalUpload/review_data_processed.csv"
data_type = AssetTypes.URI_FILE
input_mode = InputOutputModes.RO_MOUNT
output_mode = InputOutputModes.RW_MOUNT



outputs = {
    "output_data": Output(type=data_type, 
                          path=output_path, 
                          mode=output_mode,
                          name = "review-data-processed",
                  )
}

# it does not download the data, just keep a reference
# i cant manage to get the latest version automatically so i hardcoded it
input_dataset = ML_CLIENT.data.get(str(NAME_DATA_ASSET_RAW)+"-test-subset", version='2024.04.04.181358')


pipeline_job = preprocessing_pipeline(
    data=input_dataset, 
    output=outputs,
    out_path=output_path,
)
ML_CLIENT.jobs.create_or_update(pipeline_job)