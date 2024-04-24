from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class AzureMlConfig:
    subscription_id: str
    resource_group: str
    workspace_name: str
    storage_account_name: str
    key_storage: str

    @classmethod
    def load_config(cls, config_azure_ml_path: Path, config_storage:Path) -> "AzureMlConfig":
        with open(config_azure_ml_path, 'r') as config_file:
            azure_ml_config_data = json.load(config_file)
        with open(config_storage, 'r') as config_file:
            storage_config_data = json.load(config_file)
        
        combined_config_data = {**azure_ml_config_data, **storage_config_data}

        return cls(**combined_config_data)