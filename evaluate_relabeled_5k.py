"""
Evaluate Production Stacking Ensemble on Relabeled 5K Hard Dataset
"""

import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessor import HinglishPreprocessor
from src.models import EnsembleClassifier

def evaluate_models_on_relabeled_5k():
    data_path = os.path.join(PROJECT_ROOT, "data", "raw", "hinglish_hard_ambiguous_dataset_5000_relabeled.csv")
    cat_model_path = os.path.join(PROJECT_ROOT, "models", "category_ensemble.pkl")
    urg_model_path = os.path.join(PROJECT_ROOT, "models", "urgency_ensemble.pkl")

    print(f"Loading relabeled 5K dataset: {data_path}")
    df = pd.read_csv(data_path)

    preprocessor = HinglishPreprocessor()
    df['clean_text'] = df['text'].apply(preprocessor.preprocess)

    X_eval = df['clean_text'].to_numpy()
    y_cat_true = df['category'].to_numpy()
    y_urg_true = df['urgency'].to_numpy()

    # Load Category Model
    print("\nLoading Category Model...")
    cat_model = joblib.load(cat_model_path)
    y_cat_pred = cat_model.predict(X_eval)

    cat_acc = accuracy_score(y_cat_true, y_cat_pred)
    cat_f1_macro = f1_score(y_cat_true, y_cat_pred, average='macro')
    cat_f1_weighted = f1_score(y_cat_true, y_cat_pred, average='weighted')

    print("\n" + "=" * 60)
    print("  CATEGORY EVALUATION RESULTS ON 5K HARD DATASET")
    print("=" * 60)
    print(f"  Accuracy:            {cat_acc:.4f}")
    print(f"  Macro F1:            {cat_f1_macro:.4f}")
    print(f"  Weighted F1:         {cat_f1_weighted:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_cat_true, y_cat_pred))

    # Load Urgency Model
    print("\nLoading Urgency Model...")
    urg_model = joblib.load(urg_model_path)
    y_urg_pred = urg_model.predict(X_eval)

    urg_acc = accuracy_score(y_urg_true, y_urg_pred)
    urg_f1_macro = f1_score(y_urg_true, y_urg_pred, average='macro')
    urg_f1_weighted = f1_score(y_urg_true, y_urg_pred, average='weighted')

    print("\n" + "=" * 60)
    print("  URGENCY EVALUATION RESULTS ON RELABELED 5K HARD DATASET")
    print("=" * 60)
    print(f"  Accuracy:            {urg_acc:.4f}")
    print(f"  Macro F1:            {urg_f1_macro:.4f}")
    print(f"  Weighted F1:         {urg_f1_weighted:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_urg_true, y_urg_pred))

if __name__ == "__main__":
    evaluate_models_on_relabeled_5k()
