import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.train import train

def evaluate():
    X_test, y_test, xgb_model, ada_model, cat_model, le = train()

    models = {
        'XGBoost': xgb_model,
        'AdaBoost': ada_model,
        'CatBoost': cat_model
    }

    labels = le.classes_

    print("\n" + "="*50)
    print("MODEL EVALUATION REPORT")
    print("="*50)

    for name, model in models.items():
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)

        print(f"\nModel: {name}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")

    # Plot confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                    xticklabels=labels, yticklabels=labels, cmap='Blues')
        ax.set_title(name)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig('models/confusion_matrices.png')
    plt.show()
    print("\nConfusion matrices saved!")

if __name__ == "__main__":
    evaluate()