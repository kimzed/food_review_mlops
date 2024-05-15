import argparse
from typing import Callable
from pathlib import WindowsPath
import pandas as pd
from preprocessing_functions import clean_text, add_polarity_label
from centralized_preprocessing import CentralizedPreprocessing

def processing_pipeline(
    path_dataset: WindowsPath, modify_func: Callable[[pd.DataFrame], pd.DataFrame],
    pipeline: CentralizedPreprocessing
) -> pd.DataFrame:

    df = pd.read_csv(path_dataset, encoding="utf8")

    # removing duplicates
    df.drop_duplicates(subset=["Text"], inplace=True)

    # add extra polarity label
    df = modify_func(df)

    df["text_cleaned"] = df.Text.apply(lambda x: pipeline.preprocess(x))
    df["summary_cleaned"] = df.Summary.apply(lambda x: pipeline.preprocess(x))

    return df


def main() -> pd.DataFrame:

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to cloud input data")
    parser.add_argument("--output", type=str, help="path to cloud output data")
    args = parser.parse_args()

    pipeline = CentralizedPreprocessing()
    df = processing_pipeline(path_dataset=args.data, modify_func=add_polarity_label, pipeline=pipeline)

    df.to_csv(args.output, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
