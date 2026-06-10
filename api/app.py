import re
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from fastapi import FastAPI
from pydantic import BaseModel

# Load models
xgb_model = joblib.load('models/xgb_model.joblib')
vectorizer = joblib.load('models/vectorizer.joblib')
le = joblib.load('models/label_encoder.joblib')

# FastAPI app
app = FastAPI(title="10-K SEC Filing Classifier")

# Input schema
class TextInput(BaseModel):
    text: str

# Risk words
high_risk_words = [
    'risk', 'lawsuit', 'litigation', 'legal', 'court', 'plaintiff', 'defendant',
    'settlement', 'penalty', 'fine', 'violation', 'fraud', 'investigation',
    'loss', 'debt', 'default', 'bankruptcy', 'insolvency', 'deficit',
    'impairment', 'writeoff', 'restructuring', 'downgrade', 'liquidation',
    'decline', 'decrease', 'adverse', 'uncertainty', 'volatile', 'unstable',
    'unfavorable', 'deterioration', 'challenge', 'difficult', 'failure',
    'unable', 'cannot', 'threat', 'exposure', 'vulnerable', 'concern',
    'negative'
]

low_risk_words = [
    'growth', 'profit', 'revenue', 'income', 'surplus', 'dividend',
    'earnings', 'margin', 'cashflow', 'liquidity', 'solvent',
    'increase', 'success', 'strong', 'positive', 'opportunity', 'expansion',
    'improvement', 'gain', 'efficient', 'innovative', 'leading', 'competitive',
    'stable', 'consistent', 'reliable', 'favorable', 'confident',
    'record', 'exceed', 'outperform', 'milestone', 'achievement', 'momentum'
]

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z A-Z ]', '', text)
    text = text.lower()
    return text

def extract_features(text):
    cleaned = clean_text(text)

    # TF-IDF
    X_tfidf = vectorizer.transform([cleaned])

    # Custom features
    doc_length = len(cleaned.split())
    high_count = sum(cleaned.count(w) for w in high_risk_words)
    low_count = sum(cleaned.count(w) for w in low_risk_words)
    risk_ratio = high_count / (low_count + 1)

    custom = csr_matrix([[doc_length, high_count, low_count, risk_ratio]])
    return hstack([X_tfidf, custom])

@app.get("/")
def home():
    return {"message": "10-K SEC Filing Classifier is running!"}

@app.post("/predict")
def predict(input: TextInput):
    X = extract_features(input.text)
    pred = xgb_model.predict(X)[0]
    proba = xgb_model.predict_proba(X)[0]
    label = le.inverse_transform([pred])[0]
    confidence = round(float(np.max(proba)), 4)

    return {
        "label": label,
        "confidence": confidence
    }