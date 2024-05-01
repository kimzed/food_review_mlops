from src.azure_pipelines.azure_ml_setup import ML_CLIENT
from settings import NAME_DATA_ASSET_RAW, AZURE_ENV_NAME, COMPUTE_NAME, WORK_DIR
from azure.ai.ml import Input
from azure.ai.ml import Input, Output
from azure.ai.ml import command
from azure.ai.ml.constants import AssetTypes, InputOutputModes


output_path = "processed_reviews.csv"
data_type = AssetTypes.URI_FILE
input_mode = InputOutputModes.RO_MOUNT
output_mode = InputOutputModes.RW_MOUNT


outputs = {
    "output_data": Output(
        type=data_type,
        path=output_path,
        mode=output_mode,
        name="review-data-processed",
    )
}

data_asset_name = "review_data_processed"
data_type = AssetTypes.URI_FILE
output_path = "azureml://datastores/workspaceblobstore/paths/feature_pipeline_output/review_data_processed.csv"
output_mode = InputOutputModes.RW_MOUNT
outputs = {
    "output_data": Output(
        type=AssetTypes.URI_FILE,
        path=output_path,
        mode=output_mode,
        name=data_asset_name,
    )
}

# it does not download the data, just keep a reference
# i cant manage to get the latest version automatically so i hardcoded it
input_dataset = ML_CLIENT.data._get_latest_version(name=NAME_DATA_ASSET_RAW)
inputs = {"input_data": Input(type=data_type, path=input_dataset.path, mode=input_mode)}

job = command(
    command="python src/preprocessing.py --data ${{inputs.input_data}} --output ${{outputs.output_data}}",
    inputs=inputs,
    outputs=outputs,
    code=WORK_DIR,
    environment=f"{AZURE_ENV_NAME}@latest",
    compute=COMPUTE_NAME,
    display_name="preprocessing_review_data",
)
ML_CLIENT.jobs.create_or_update(job)
