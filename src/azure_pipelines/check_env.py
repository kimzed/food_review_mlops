from azureml.core import Workspace, Environment
from settings import AZURE_ML_CONFIG_FILE, AZURE_ENV_NAME
from src.azure_pipelines.azure_ml_setup import ML_WORKSPACE

# # Load your Azure ML Workspace
# ws = Workspace.from_config(path=AZURE_ML_CONFIG_FILE)

# # List all environments in the workspace
# envs = Environment.list(workspace=ws)
# for name, env in envs.items():
#     print(name)

# # Get a specific environment by name
# my_env = Environment.get(workspace=ws, name=AZURE_ENV_NAME)
# print(my_env)


from azureml.core import Workspace


# List all environments in the workspace
env_names = Environment.list(workspace=ML_WORKSPACE)
print(env_names)
