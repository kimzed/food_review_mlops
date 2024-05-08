from pathlib import Path
from src.azure_pipelines.azure_ml_config import AzureMlConfig

WORK_DIR = Path(__file__).parent
DATA_DIR = WORK_DIR / "data/"
DATA_DIR.mkdir(exist_ok=True)

REVIEW_FILE_PATH = DATA_DIR / "Reviews.csv"

CONFIG_DIR = WORK_DIR / "config/"
AZURE_ML_CONFIG_FILE = CONFIG_DIR / "config_azure_ml.json"

CONFIG_AZURE_ML = AzureMlConfig.load_config(config_azure_ml_path=AZURE_ML_CONFIG_FILE)

# we use a subset for now to speed up the calculation
NAME_DATA_ASSET_RAW = "amazon-reviews-raw-test-subset"
NAME_DATA_ASSET_FEATURE = "review_data_processed"
MODEL_NAME = "sentiment_analysis_model"

AZURE_ENV_FILE = WORK_DIR / "environment.yml"

AZURE_ENV_NAME = "food-review"

COMPUTE_NAME = "aa-capstone"
