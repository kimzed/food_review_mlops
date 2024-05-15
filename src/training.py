import argparse
import os
import mlflow
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from training_functions import apply_sentiment_analysis
import pandas as pd
import numpy as np
import xgboost as xgb
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

from centralized_preprocessing import CentralizedPreprocessing
import joblib

def xgb_classifier(vect: CountVectorizer, df: pd.DataFrame) -> np.ndarray:
    x_text_features = vect.fit_transform(df.combined_text_Summary)
    df[[-1]] = df[["comb_sentiment"]]
    x_sentiment = df[[-1]]
    x = pd.concat(
        [pd.DataFrame(x_sentiment), pd.DataFrame(x_text_features.toarray())], axis=1
    )
    y = df["Score"] - 1
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    model = xgb.XGBClassifier(objective="multi:softmax", num_class=5, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = classification_report(y_test, predictions, output_dict=True)

    mlflow.log_metric("accuracy", metrics.pop("accuracy"))
    for class_or_avg, metrics_dict in metrics.items():
        for metric, value in metrics_dict.items():
            mlflow.log_metric(class_or_avg + "_" + metric, value)

    mlflow.end_run()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to cloud input data")
    parser.add_argument("--output", type=str, help="path to cloud output data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.2)
    parser.add_argument(
        "--registered_model_name",
        type=str,
        help="Name of the model to be saved in the registery",
    )
    args = parser.parse_args()

    mlflow.start_run()
    mlflow.sklearn.autolog()

    df = pd.read_csv(args.data)

    # Apply the sentiment analysis function to the DataFrame
    df = apply_sentiment_analysis(df)

    pipeline = CentralizedPreprocessing()

    x_vectorized = pipeline.fit_transform(df.combined_text_Summary)

    mlflow.log_metric("num_samples", x_vectorized.shape[0])
    mlflow.log_metric("num_features", x_vectorized.shape[1] - 1)

    x_sentiment = df[["comb_sentiment"]]
    x = pd.concat(
        [pd.DataFrame(x_sentiment), pd.DataFrame(x_vectorized.toarray())], axis=1
    )
    y = df["Score"] - 1  # class needs to start by 0
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_train_ratio, random_state=42
    )
    model = xgb.XGBClassifier(objective="multi:softmax", num_class=5, random_state=42)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = classification_report(y_test, predictions, output_dict=True)

    mlflow.log_metric("accuracy", metrics.pop("accuracy"))
    for class_or_avg, metrics_dict in metrics.items():
        for metric, value in metrics_dict.items():
            mlflow.log_metric(class_or_avg + "_" + metric, value)

    print("Registering the models via MLFlow")
    mlflow.xgboost.log_model(
        xgb_model=model,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
    )

    mlflow.pyfunc.log_model(python_model=pipeline, registered_model_name="preprocessing_pipeline",
                            artifact_path="preprocessing_pipeline")


if __name__ == "__main__":
    main()
