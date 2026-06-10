import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier

from src.preprocess import load_data, preprocess
from src.features import engineer_features

def train():
    # Load and preprocess
    print("Loading data...")
    df = load_data()
    df = preprocess(df)
    print("Preprocessing done!")

    # Feature engineering
    print("Engineering features...")
    X, y_raw, vectorizer = engineer_features(df)
    print("Features done!")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

    # Train models
    print("\nTraining XGBoost...")
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)
    print("XGBoost done!")

    print("Training AdaBoost...")
    ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
    ada_model.fit(X_train, y_train)
    print("AdaBoost done!")

    print("Training CatBoost...")
    cat_model = CatBoostClassifier(iterations=100, random_seed=42, verbose=0)
    cat_model.fit(X_train, y_train)
    print("CatBoost done!")

    # Save models
    joblib.dump(xgb_model, 'models/xgb_model.joblib')
    joblib.dump(vectorizer, 'models/vectorizer.joblib')
    joblib.dump(le, 'models/label_encoder.joblib')
    print("\nModels saved to models/ folder!")

    return X_test, y_test, xgb_model, ada_model, cat_model, le

if __name__ == "__main__":
    train()