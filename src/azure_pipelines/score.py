import os
import logging
import json
import numpy
import joblib
from settings import MODEL_NAME

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    global pipeline
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    model_dir = os.getenv("AZUREML_MODEL_DIR")

    model_path = os.path.join(
        model_dir, MODEL_NAME
    )
    pipeline_path = os.path.join(
        model_dir, 'text_pipeline.pkl'
    )
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)

    # load text pipeline
    pipeline = joblib.load(pipeline_path)

    logging.info("Init complete")

def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("Request received")
    return "prediction as list"