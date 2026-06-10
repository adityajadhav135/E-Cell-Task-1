import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

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

def assign_label(text):
    high_score = sum(text.count(word) for word in high_risk_words)
    low_score = sum(text.count(word) for word in low_risk_words)

    if high_score > low_score * 1.1:
        return 'high'
    elif low_score > high_score * 1.1:
        return 'low'
    else:
        return 'medium'

def engineer_features(df):
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(df['clean_text'])

    # Custom features
    df['doc_length'] = df['clean_text'].apply(lambda x: len(x.split()))
    df['high_risk_count'] = df['clean_text'].apply(lambda x: sum(x.count(w) for w in high_risk_words))
    df['low_risk_count'] = df['clean_text'].apply(lambda x: sum(x.count(w) for w in low_risk_words))
    df['risk_ratio'] = df['high_risk_count'] / (df['low_risk_count'] + 1)

    # Combine
    custom = csr_matrix(df[['doc_length', 'high_risk_count', 'low_risk_count', 'risk_ratio']].values)
    X_combined = hstack([X_tfidf, custom])

    # Labels
    df['label'] = df['clean_text'].apply(assign_label)

    return X_combined, df['label'], vectorizer

if __name__ == "__main__":
    print("Features module ready!")