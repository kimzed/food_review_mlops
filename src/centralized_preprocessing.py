import argparse
from typing import Callable
from pathlib import WindowsPath
import pandas as pd
from preprocessing_functions import clean_text, add_polarity_label

from training_functions import apply_sentiment_analysis

from sklearn.feature_extraction.text import CountVectorizer

class CentralizedPreprocessing:
    def __init__(self):
        self.vectorizer = CountVectorizer(
        ngram_range=(1, 2), max_features=10000, min_df=2, stop_words="english"
    )
        self.fitted = False

    def preprocess(self, text):
        return clean_text(text)

    def fit_transform(self, texts):
        preprocessed_texts = [self.preprocess(text) for text in texts]
        self.fitted = True
        return self.vectorizer.fit_transform(preprocessed_texts)

    def transform(self, text):
        if not self.fitted:
            raise RuntimeError("The pipeline needs to be fitted before transforming data.")
        preprocessed_text = self.preprocess(text)
        return self.vectorizer.transform([preprocessed_text])