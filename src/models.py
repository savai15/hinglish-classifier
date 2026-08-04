"""
Model Definitions for Hinglish Complaint Classification (Enhanced v2)

Provides:
1. TF-IDF + Logistic Regression baseline
2. Character n-gram + Logistic Regression (mimics FastText subword advantage)
3. Combined word + character n-gram pipeline
4. TF-IDF + Linear SVM pipeline
5. Hyperparameter tuning with RandomizedSearchCV
"""
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform


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


def get_param_grid(pipeline_type):
    """
    Get hyperparameter search space for a given pipeline type.

    Parameters
    ----------
    pipeline_type : str
        One of 'tfidf', 'char_ngram', 'combined', 'svm'

    Returns
    -------
    dict
        Parameter distribution for RandomizedSearchCV
    """
    if pipeline_type == 'tfidf':
        return {
            'tfidf__max_features': [8000, 10000, 15000, 20000, 25000],
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'tfidf__min_df': [1, 2, 3],
            'tfidf__max_df': [0.90, 0.95, 1.0],
            'classifier__C': loguniform(1e-2, 1e2),
        }
    elif pipeline_type == 'char_ngram':
        return {
            'tfidf__max_features': [20000, 30000, 40000, 50000],
            'tfidf__ngram_range': [(2, 4), (2, 5), (2, 6), (3, 5)],
            'tfidf__min_df': [1, 2, 3],
            'tfidf__max_df': [0.90, 0.95, 1.0],
            'classifier__C': loguniform(1e-2, 1e2),
        }
    elif pipeline_type == 'combined':
        return {
            'features__word_tfidf__max_features': [10000, 15000, 20000],
            'features__word_tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'features__char_tfidf__max_features': [20000, 30000, 40000],
            'features__char_tfidf__ngram_range': [(2, 4), (2, 5), (2, 6)],
            'classifier__C': loguniform(1e-2, 1e2),
        }
    elif pipeline_type == 'svm':
        return {
            'tfidf__max_features': [8000, 10000, 15000, 20000, 25000],
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'tfidf__min_df': [1, 2, 3],
            'tfidf__max_df': [0.90, 0.95, 1.0],
            'classifier__estimator__C': loguniform(1e-2, 1e2),
        }
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")


def tune_model(pipeline, param_grid, X_train, y_train, n_iter=30, cv=3,
               scoring='f1_macro', random_state=42, verbose=1):
    """
    Tune a pipeline's hyperparameters using RandomizedSearchCV.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The model pipeline to tune
    param_grid : dict
        Parameter search space
    X_train : array-like
        Training text data
    y_train : array-like
        Training labels
    n_iter : int
        Number of random combinations to try
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric
    random_state : int
        Random seed
    verbose : int
        Verbosity level

    Returns
    -------
    tuple
        (best_pipeline, search_results) - fitted best model and search results
    """
    search = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1,
        verbose=verbose,
        error_score=0.0
    )

    search.fit(X_train, y_train)

    return search.best_estimator_, search


def get_pipeline_type(pipeline):
    """Detect the pipeline type from its step names and classifier type."""
    step_names = [name for name, _ in pipeline.steps]
    if 'features' in step_names:
        return 'combined'
    elif 'tfidf' in step_names:
        # Check if classifier is CalibratedClassifierCV (SVM) or LogisticRegression
        classifier = pipeline.named_steps.get('classifier')
        if classifier is not None:
            cls_type = type(classifier).__name__
            if cls_type == 'CalibratedClassifierCV':
                return 'svm'
        # Check if it's char or word
        tfidf = pipeline.named_steps.get('tfidf')
        if tfidf and tfidf.analyzer == 'char_wb':
            return 'char_ngram'
        return 'tfidf'
    return 'unknown'


def train_model(pipeline, X_train, y_train, verbose=True):
    """
    Train a pipeline on the given data.
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
