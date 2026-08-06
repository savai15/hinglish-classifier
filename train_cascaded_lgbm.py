"""
Train and Evaluate Cascaded LightGBM GBDT Pipeline
"""

import os
import sys
import time
import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessor import HinglishPreprocessor
from src.cascaded_lgbm import CascadedLGBMClassifier

def run_cascaded_lgbm_pipeline():
    print("=" * 70)
    print("  CASCADED LIGHTGBM GBDT PIPELINE")
    print("  Stage 1: 10-Category LightGBM GBDT")
    print("  Stage 2: Category-Conditioned LightGBM Urgency GBDT")
    print("=" * 70)

    # 1. Load Data
    data_50k_path = os.path.join(PROJECT_ROOT, "data", "raw", "hinglish_dataset_50000_v2.csv")
    hard_5k_path = os.path.join(PROJECT_ROOT, "data", "raw", "hinglish_hard_ambiguous_dataset_5000_relabeled.csv")

    print(f"\n[1/4] Loading 50K Dataset: {data_50k_path}")
    df_50k = pd.read_csv(data_50k_path)

    preprocessor = HinglishPreprocessor()
    df_50k['clean_text'] = df_50k['text'].apply(preprocessor.preprocess)

    X = df_50k['clean_text'].to_numpy()
    y_cat = df_50k['category'].to_numpy()
    y_urg = df_50k['urgency'].to_numpy()

    X_train, X_test, y_cat_train, y_cat_test, y_urg_train, y_urg_test = train_test_split(
        X, y_cat, y_urg, test_size=0.15, random_state=42, stratify=y_cat
    )
    print(f"  Train samples: {len(X_train)} | Held-out test samples: {len(X_test)}")

    # 2. Train Cascaded Model
    print("\n[2/4] Fitting Cascaded LightGBM GBDT Model...")
    start_time = time.time()
    cascaded_model = CascadedLGBMClassifier(n_estimators=120, learning_rate=0.08)
    cascaded_model.fit(X_train, y_cat_train, y_urg_train)
    elapsed = time.time() - start_time
    print(f"  Model training completed in {elapsed:.1f}s!")

    # Save model
    model_save_path = os.path.join(PROJECT_ROOT, "models", "cascaded_lgbm_model.pkl")
    joblib.dump(cascaded_model, model_save_path)
    print(f"  Saved trained model to {model_save_path}")

    # 3. Evaluate on Held-Out 50K Test Set
    print("\n[3/4] Evaluating on Held-Out Test Set (7,500 samples)...")
    cat_preds, urg_preds = cascaded_model.predict(X_test)

    print("\n" + "=" * 50)
    print("  HELD-OUT TEST SET - CATEGORY PERFORMANCE")
    print("=" * 50)
    print(f"  Accuracy:    {accuracy_score(y_cat_test, cat_preds):.4f}")
    print(f"  Macro F1:    {f1_score(y_cat_test, cat_preds, average='macro'):.4f}")

    print("\n" + "=" * 50)
    print("  HELD-OUT TEST SET - URGENCY PERFORMANCE")
    print("=" * 50)
    print(f"  Accuracy:    {accuracy_score(y_urg_test, urg_preds):.4f}")
    print(f"  Macro F1:    {f1_score(y_urg_test, urg_preds, average='macro'):.4f}")

    # 4. Evaluate on Relabeled 5K Hard Dataset
    print("\n[4/4] Evaluating on Relabeled 5K Hard Dataset...")
    df_5k = pd.read_csv(hard_5k_path)
    df_5k['clean_text'] = df_5k['text'].apply(preprocessor.preprocess)

    X_5k = df_5k['clean_text'].to_numpy()
    y_cat_5k = df_5k['category'].to_numpy()
    y_urg_5k = df_5k['urgency'].to_numpy()

    cat_preds_5k, urg_preds_5k = cascaded_model.predict(X_5k)

    print("\n" + "=" * 60)
    print("  5K HARD DATASET - CATEGORY PERFORMANCE (LightGBM)")
    print("=" * 60)
    print(f"  Accuracy:       {accuracy_score(y_cat_5k, cat_preds_5k):.4f}")
    print(f"  Weighted F1:    {f1_score(y_cat_5k, cat_preds_5k, average='weighted'):.4f}")
    print(f"  Macro F1:       {f1_score(y_cat_5k, cat_preds_5k, average='macro'):.4f}")

    print("\n" + "=" * 60)
    print("  5K HARD DATASET - URGENCY PERFORMANCE (Cascaded LightGBM)")
    print("=" * 60)
    print(f"  Accuracy:       {accuracy_score(y_urg_5k, urg_preds_5k):.4f}")
    print(f"  Weighted F1:    {f1_score(y_urg_5k, urg_preds_5k, average='weighted'):.4f}")
    print(f"  Macro F1:       {f1_score(y_urg_5k, urg_preds_5k, average='macro'):.4f}")
    print("\nClassification Report (Urgency):")
    print(classification_report(y_urg_5k, urg_preds_5k))

if __name__ == "__main__":
    run_cascaded_lgbm_pipeline()
