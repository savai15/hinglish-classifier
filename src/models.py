"""
Model Definitions for Hinglish Complaint Classification

Provides:
1. TF-IDF + Logistic Regression baseline
2. Character n-gram + Logistic Regression (mimics FastText subword advantage)
3. Helper functions for training, predicting, and saving/loading models
"""
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV


def build_tfidf_pipeline(
    max_features=15000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    C=1.0,
    class_weight='balanced',
    random_state=42
):
    """
    Build a TF-IDF + Logistic Regression pipeline (baseline approach).

    Uses word-level TF-IDF features with unigrams and bigrams.
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
            analyzer='word',
            strip_accents=None,
            token_pattern=r'(?u)\b\w+\b'
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,
            C=C,
            class_weight=class_weight,
            random_state=random_state,
            solver='lbfgs'
        ))
    ])
    return pipeline


def build_char_ngram_pipeline(
    max_features=30000,
    ngram_range=(2, 5),
    min_df=2,
    max_df=0.95,
    C=1.0,
    class_weight='balanced',
    random_state=42
):
    """
    Build a Character N-gram + Logistic Regression pipeline.

    This mimics FastText's subword advantage by using character n-grams.
    Character n-grams naturally handle:
    - Spelling variants (nahi/nai/nahee share character n-grams)
    - Out-of-vocabulary words
    - Code-mixed Hindi-English text
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
            analyzer='char_wb',
            strip_accents=None
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,
            C=C,
            class_weight=class_weight,
            random_state=random_state,
            solver='lbfgs'
        ))
    ])
    return pipeline


def build_combined_pipeline(
    max_features_word=15000,
    max_features_char=30000,
    word_ngram_range=(1, 2),
    char_ngram_range=(2, 5),
    C=1.0,
    class_weight='balanced',
    random_state=42
):
    """
    Build a combined word + character n-gram pipeline.

    Uses both word-level and character-level features concatenated.
    This combines the best of both approaches.
    """
    from sklearn.pipeline import FeatureUnion

    combined = FeatureUnion([
        ('word_tfidf', TfidfVectorizer(
            max_features=max_features_word,
            ngram_range=word_ngram_range,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            analyzer='word',
            strip_accents=None,
            token_pattern=r'(?u)\b\w+\b'
        )),
        ('char_tfidf', TfidfVectorizer(
            max_features=max_features_char,
            ngram_range=char_ngram_range,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            analyzer='char_wb',
            strip_accents=None
        ))
    ])

    pipeline = Pipeline([
        ('features', combined),
        ('classifier', LogisticRegression(
            max_iter=1000,
            C=C,
            class_weight=class_weight,
            random_state=random_state,
            solver='lbfgs'
        ))
    ])
    return pipeline


def build_svm_pipeline(
    max_features=15000,
    ngram_range=(1, 2),
    C=1.0,
    class_weight='balanced',
    random_state=42
):
    """
    Build a TF-IDF + Linear SVM pipeline.

    Linear SVM often outperforms Logistic Regression on text classification.
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            analyzer='word',
            strip_accents=None,
            token_pattern=r'(?u)\b\w+\b'
        )),
        ('classifier', CalibratedClassifierCV(
            LinearSVC(
                max_iter=2000,
                C=C,
                class_weight=class_weight,
                random_state=random_state
            ),
            cv=3
        ))
    ])
    return pipeline


def train_model(pipeline, X_train, y_train, verbose=True):
    """
    Train a pipeline on the given data.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The model pipeline
    X_train : array-like
        Training text data
    y_train : array-like
        Training labels
    verbose : bool
        If True, print training info

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        The trained pipeline
    """
    if verbose:
        print(f"  Training on {len(X_train)} samples...")

    pipeline.fit(X_train, y_train)

    if verbose:
        print("  Training complete.")

    return pipeline


def predict(pipeline, X):
    """Get predictions from a trained pipeline."""
    return pipeline.predict(X)


def predict_proba(pipeline, X):
    """Get probability predictions from a trained pipeline."""
    if hasattr(pipeline, 'predict_proba'):
        return pipeline.predict_proba(X)
    else:
        return None


def save_model(pipeline, filepath):
    """Save a trained pipeline to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"  Model saved to {filepath}")


def load_model(filepath):
    """Load a trained pipeline from disk."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# Alias for backward compatibility
FeatureUnion = None
try:
    from sklearn.pipeline import FeatureUnion
except ImportError:
    pass
