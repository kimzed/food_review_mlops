from azure.ai.ml import command
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from src.azure_pipelines.azure_ml_setup import ML_CLIENT
from settings import AZURE_ENV_NAME, COMPUTE_NAME, MODEL_NAME, NAME_DATA_ASSET_FEATURE, WORK_DIR


def main():
    input_mode = InputOutputModes.RO_MOUNT
    data_type = AssetTypes.URI_FILE
    input_dataset = ML_CLIENT.data._get_latest_version(name=NAME_DATA_ASSET_FEATURE)

    inputs = Input(type=data_type, path=input_dataset.path, mode=input_mode)

    job = command(
        inputs=dict(
            data=inputs,
            test_train_ratio=0.2,
            registered_model_name=MODEL_NAME,
        ),
        code=WORK_DIR,  # location of source code
        command="python src/training.py --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} --registered_model_name ${{inputs.registered_model_name}}",
        environment=f"{AZURE_ENV_NAME}@latest",
        compute=COMPUTE_NAME,
        display_name="training_sentiment_analysis",
    )
    ML_CLIENT.jobs.create_or_update(job)

if __name__ == "__main__":
    main()