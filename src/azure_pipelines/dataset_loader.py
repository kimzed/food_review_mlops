import time
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from settings import REVIEW_FILE_PATH, NAME_DATA_ASSET_RAW
from src.azure_pipelines.azure_ml_setup import ML_CLIENT
import pandas as pd  # You need to import pandas to handle CSV files


def main():
    dataset_version = time.strftime("%Y.%m.%d.%H%M%S", time.gmtime())

    # Step 1: Load the original CSV file using pandas
    df = pd.read_csv(REVIEW_FILE_PATH)

    # TODO remove when the pipeline is finished
    df_subset = df.head(100)
    df_subset.to_csv(REVIEW_FILE_PATH, index=False)

    my_data = Data(
        name=NAME_DATA_ASSET_RAW,
        version=dataset_version,
        description="Unprocessed dataset of amazon reviews (first 100 lines)",
        path=REVIEW_FILE_PATH,  # Use the new file path
        type=AssetTypes.URI_FILE,
    )

    ML_CLIENT.data.create_or_update(my_data)

    print(f"Data asset created. Name: {my_data.name}, version: {my_data.version}")


if __name__ == "__main__":
    main()
