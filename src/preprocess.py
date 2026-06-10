import re
import pandas as pd
from datasets import load_dataset

def load_data():
    dataset = load_dataset(
        "winterForestStump/10-K_sec_filings",
        streaming=True
    )
    df = pd.DataFrame(list(dataset["001"].take(2000)))
    return df

def extract_text(df):
    text_columns = [
        'Business',
        'Management’s Discussion and Analysis of Financial Condition and Results of Operations',
        'Legal Proceedings'
    ]
    df['raw_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)
    return df

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z A-Z ]', '', text)
    text = text.lower()
    return text

def preprocess(df):
    df = extract_text(df)
    df['clean_text'] = df['raw_text'].apply(clean_text)
    return df

if __name__ == "__main__":
    df = load_data()
    df = preprocess(df)
    print(df[['clean_text']].head())
    print("Preprocessing done!")
