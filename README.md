# Food Review MLOps

> **Work in progress**

End-to-end ML pipeline for sentiment analysis on food reviews, with CI/CD automation and Azure ML integration.

## Overview

An MLOps project that implements a complete machine learning pipeline for multi-class sentiment prediction (1–5 star ratings) on Amazon food reviews. The pipeline covers text preprocessing, NLP feature engineering, XGBoost model training, and deployment via Azure ML, with automated testing through GitHub Actions.

## Tech Stack

- **ML:** XGBoost, scikit-learn, spaCy, NLTK, TextBlob
- **Experiment Tracking:** MLflow
- **Cloud:** Azure ML (pipelines, model registry, endpoints)
- **CI/CD:** GitHub Actions (automated tests on PR)
- **Infrastructure:** Azure Bicep (IaC)
- **Environment:** Conda (Python 3.11)
- **Testing:** pytest

## Pipeline

```
Raw Reviews CSV
    ↓
Preprocessing (HTML removal, lemmatization, contraction expansion)
    ↓
Feature Engineering (CountVectorizer n-grams + TextBlob sentiment)
    ↓
XGBoost Multi-Class Training (5 classes, MLflow logging)
    ↓
Model Registry → Azure ML Endpoint
```

## Features

- **Text Preprocessing** — HTML removal, accent normalization, contraction expansion, regex cleaning, spaCy lemmatization, negation-aware stop word removal
- **Feature Extraction** — Count vectorization with n-grams (1-2) combined with sentiment polarity scores
- **Model Training** — XGBoost multi-class classifier with per-class precision/recall/F1 logging via MLflow
- **CI/CD Pipeline** — GitHub Actions runs pytest on PRs to `dev` branch
- **Azure ML Integration** — Dataset versioning, compute cluster management, model registration, online endpoint deployment

## Project Structure

```
food_review_mlops/
├── src/
│   ├── preprocessing.py               # Text cleaning pipeline
│   ├── preprocessing_functions.py      # Modular preprocessing functions
│   ├── training.py                     # XGBoost training with MLflow
│   ├── training_functions.py           # Sentiment analysis helpers
│   └── azure_pipelines/               # Azure ML orchestration
│       ├── dataset_loader.py           # Data asset management
│       ├── feature_pipeline.py         # Feature engineering job
│       ├── training_pipeline.py        # Training job
│       ├── inference_pipeline.py       # Deployment config
│       └── score.py                    # Scoring script
├── tests/
│   └── test_preprocessing_functions.py # Unit tests
├── deploy/
│   └── main.bicep                      # Azure infrastructure
├── .github/workflows/
│   └── main.yml                        # CI pipeline
├── environment.yml                     # Conda environment
└── settings.py                         # Configuration
```

## Getting Started

```bash
conda env create -f environment.yml
conda activate food-review
python -m spacy download en_core_web_sm
pytest tests/
```

## Context

Course assignment completed during a traineeship, demonstrating ML engineering practices including experiment tracking, cloud deployment, infrastructure as code, and automated testing.
