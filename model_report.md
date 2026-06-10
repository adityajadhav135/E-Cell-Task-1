# Model Report — 10-K SEC Filing Classifier

## What are we trying to classify?
We classify 10-K SEC filings into three financial risk levels — **high, medium, or low risk**.

Since the dataset has no pre-made labels, we created them using keyword scoring inspired by the Loughran-McDonald Financial Sentiment Dictionary, a well known word list used in finance research.

- **High Risk** — the filing has a lot more risk/negative words than positive words
- **Low Risk** — the filing has a lot more positive/growth words than risk words
- **Medium Risk** — the filing has a roughly equal mix of both

## How did we clean the data?

We used three sections from each filing:
- Business Overview
- Management Discussion and Analysis (MD&A)
- Legal Proceedings

We skipped Financial Statements because they are full of numbers and tables which confuse NLP models. We also skipped Risk Factors because only 8 out of 2000 rows had any data in that column.

Cleaning steps:
- Removed extra spaces and newlines
- Removed numbers, symbols and punctuation
- Converted everything to lowercase

## What features did we use?

| Feature | Why we used it |
|---------|---------------|
| TF-IDF (5000 words) | Captures which words are important in each document |
| Document length | Longer filings may have more disclosures |
| High risk word count | Direct count of risk related words |
| Low risk word count | Direct count of positive words |
| Risk ratio | Balance between risk and positive language |

We used TF-IDF as the main feature because it works really well with boosting models and is easy to explain. The custom features add financial domain knowledge on top.

## How did the models perform?

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| XGBoost | 98.75% | 98.76% | 98.75% | 98.75% |
| CatBoost | 97.50% | 97.52% | 97.50% | 97.49% |
| AdaBoost | 96.00% | 96.10% | 96.00% | 95.95% |

## Which model is the best and why?

**XGBoost** is the best model.

It got the highest score across every metric and only made 5 mistakes out of 400 test samples. Most importantly it classified all 86 high risk filings correctly without a single error.

AdaBoost had the most trouble with medium risk filings — it misclassified 14 of them. This makes sense because medium risk filings sit between high and low, making them the hardest to classify. XGBoost handled this the best because its built in regularization stops it from overfitting.

## Why do most mistakes happen in the medium class?

Medium risk filings contain a mix of both positive and negative language. This makes them harder to separate from high and low risk filings. All three models struggled most with this class, but XGBoost handled it the best.