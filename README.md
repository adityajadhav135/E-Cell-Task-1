# 10-K SEC Filing Classifier

An end-to-end document intelligence and classification system built on 10-K SEC filings from the Hugging Face dataset.

# Problem Statement
Financial reports contain critical information that is slow and inefficient to analyze manually. This system automatically classifies SEC filings into **high, medium, or low financial risk** categories.

# Dataset Used
[winterForestStump/10-K_sec_filings](https://huggingface.co/datasets/winterForestStump/10-K_sec_filings)

# Project Structure
project/
├── data/
├── notebooks/        
├── src/
│   ├── preprocess.py   
│   ├── features.py     
│   ├── train.py        
│   ├── evaluate.py     
│   └── utils.py        
├── api/
│   └── app.py       
├── models/            
├── README.md
└── requirements.txt

# Pipeline
Stage 1 — Load dataset, extract Business, MD&A, Legal Proceedings sections, clean text
Stage 2 — TF-IDF vectorization (5000 features) + custom risk features
Stage 3 — Train XGBoost, AdaBoost, CatBoost
Stage 4 — Evaluate all three models with accuracy, precision, recall, F1, confusion matrix
Stage 5 — Deploy best model via FastAPI

## Results
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| XGBoost | 98.75% | 98.75% |
| CatBoost | 97.50% | 97.49% |
| AdaBoost | 96.00% | 95.95% |

Best Model: XGBoost

# Installation
pip install -r requirements.txt

# How to Run

## Train models
python -m src.train

## Start API
python -m uvicorn api.app:app --reload

# API Endpoint
POST http://127.0.0.1:8000/predict

Input:  { "text": "..." }

Output: { "label": "high/medium/low", "confidence": 0.xx }

Swagger docs: http://127.0.0.1:8000/docs