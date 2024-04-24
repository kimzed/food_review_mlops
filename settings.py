from pathlib import Path
from src.azure_pipelines.azure_ml_config import AzureMlConfig

DATA_DIR = Path(__file__).parent /'data/'
DATA_DIR.mkdir(exist_ok=True)

REVIEW_FILE_PATH = DATA_DIR / 'Reviews.csv'

CONFIG_DIR = Path(__file__).parent /'config/'
AZURE_ML_CONFIG_FILE = CONFIG_DIR / 'config_azure_ml.json'
STORAGE_CONFIG_FILE = CONFIG_DIR / 'config_storage_account.json'
CONFIG_AZURE_ML = AzureMlConfig.load_config(config_azure_ml_path=AZURE_ML_CONFIG_FILE,
                                            config_storage=STORAGE_CONFIG_FILE)

NAME_DATA_ASSET_RAW = "amazon-reviews-raw"

AZURE_ENV_FILE = Path(__file__).parent / 'azure_environment.yml'

AZURE_ENV_NAME = "food-review"

# we had to create an environment in the website because of technical issues so we overwrite the name
AZURE_ENV_NAME = "food_review_manual_conda"

COMPUTE_NAME = "aa-capstone"