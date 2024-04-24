
import os
import argparse
from typing import Callable
from pathlib import WindowsPath
import pandas as pd
from preprocessing_functions import clean_text, add_polarity_label
from azure.storage.blob import BlobServiceClient

def processing_pipeline(path_dataset: WindowsPath,
                        modify_func: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:

    df = pd.read_csv(path_dataset,
                     encoding='utf8')
    
    # removing duplicates
    df.drop_duplicates(subset=['Text'], inplace=True)

    # add extra polarity label
    df = modify_func(df)
    
    df['text_cleaned'] = df.Text.apply(lambda x: clean_text(x))
    df['summary_cleaned'] = df.Summary.apply(lambda x: clean_text(x))

    return df

    
def main() -> pd.DataFrame:
    # this is needed for the cloud integration
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to cloud input data")
    parser.add_argument("--output", type=str, help="path to cloud output data")
    args = parser.parse_args()
    
    df = processing_pipeline(path_dataset=args.data,
                                   modify_func=add_polarity_label)
    
    blob_service = BlobServiceClient(account_url="https://aacapstone8288068060.blob.core.windows.net/", credential="ylgvzljiQ9gKmKBPw77ZvkrAH4dVvuJZocpUKald98xR5nqnLIMsaBWuttHKXp0uuNQ+O0GmKGMf+AStSKXjHA==")
    container_name = "azureml-blobstore-a16f652b-acfb-4f65-b97a-8830289e9c97"
    csv_string = df.to_csv(index=False, encoding='utf-8')
    csv_bytes = str.encode(csv_string)
    blob_client = blob_service.get_blob_client(container=container_name, blob="processed_reviews.csv")
    blob_client.upload_blob(csv_bytes, overwrite=True)

if __name__ == '__main__':
    main()