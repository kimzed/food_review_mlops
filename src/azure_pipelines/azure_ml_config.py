from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class AzureMlConfig:
    subscription_id: str
    resource_group: str
    workspace_name: str

    @classmethod
    def load_config(
        cls,
        config_azure_ml_path: Path,
    ) -> "AzureMlConfig":
        with open(config_azure_ml_path, "r") as config_file:
            azure_ml_config_data = json.load(config_file)
        return cls(**azure_ml_config_data)
