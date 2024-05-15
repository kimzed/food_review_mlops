import os
import logging
import json
import mlflow
import numpy
import joblib
import pandas as pd
from settings import MODEL_NAME

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global preprocessing_pipeline, xgb_model

    # Load the preprocessing pipeline
    preprocessing_pipeline = mlflow.pyfunc.load_model("preprocessing_pipeline")

    # Load the XGBoost model
    xgb_model = mlflow.xgboost.load_model("registered_model_name")

def run(raw_data):
    try:
        # Parse the raw data as JSON
        data = json.loads(raw_data)

        # Assuming your input data is in the form of a dictionary
        input_data = pd.DataFrame(data, index=[0])

        # Apply preprocessing pipeline
        preprocessed_data = preprocessing_pipeline.transform(input_data)

        # Make predictions using the XGBoost model
        predictions = xgb_model.predict(preprocessed_data)

        # Return the predictions as JSON
        return json.dumps(predictions.tolist())

    except Exception as e:
        error = str(e)
        return error