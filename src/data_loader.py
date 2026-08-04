"""
Data Loader for Hinglish E-Commerce Complaints (Enhanced v2)
Loads, validates, splits, and provides cross-validation for the complaint dataset.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


def load_data(csv_path):
    """
    Load the Hinglish complaints CSV dataset.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file with columns: text, category, urgency

    Returns
    -------
    pd.DataFrame
        Loaded and validated dataframe
    """
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = ['text', 'category', 'urgency']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'. Found columns: {df.columns.tolist()}")

    # Drop rows with missing text
    initial_len = len(df)
    df = df.dropna(subset=['text', 'category', 'urgency']).reset_index(drop=True)

    # Strip whitespace from text
    df['text'] = df['text'].astype(str).str.strip()

    # Remove empty texts
    df = df[df['text'].str.len() > 0].reset_index(drop=True)

    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing/empty text.")

    print(f"  Loaded {len(df)} complaints with {df['category'].nunique()} categories "
          f"and {df['urgency'].nunique()} urgency levels.")

    return df


def get_data_stats(df):
    """
    Return summary statistics of the dataset.

    Returns
    -------
    dict
        Dictionary with category counts, urgency counts, text length stats, etc.
    """
    stats = {
        'total_samples': len(df),
        'categories': df['category'].value_counts().to_dict(),
        'urgency_levels': df['urgency'].value_counts().to_dict(),
        'num_categories': df['category'].nunique(),
        'num_urgency_levels': df['urgency'].nunique(),
        'avg_text_length': df['text'].apply(len).mean(),
        'avg_word_count': df['text'].apply(lambda x: len(x.split())).mean(),
        'min_text_length': df['text'].apply(len).min(),
        'max_text_length': df['text'].apply(len).max(),
    }
    return stats


def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into train and test sets with stratification.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    test_size : float
        Fraction for test set
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        (df_train, df_test) - two DataFrames
    """
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['category']
    )

    print(f"  Train: {len(df_train)} | Test: {len(df_test)}")

    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def get_cv_splits(df, n_splits=5, random_state=42, label_column='category'):
    """
    Generate stratified k-fold cross-validation splits.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    n_splits : int
        Number of CV folds
    random_state : int
        Random seed for reproducibility
    label_column : str
        Column to stratify on

    Yields
    ------
    tuple
        (train_idx, val_idx) for each fold
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    X = df['clean_text'].values
    y = df[label_column].values

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        yield fold_idx, train_idx, val_idx


def get_cv_splits_multi(df, n_splits=5, random_state=42):
    """
    Generate CV splits that work for both category and urgency tasks.
    Uses category for stratification (primary task).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with 'clean_text', 'category', 'urgency' columns
    n_splits : int
        Number of CV folds
    random_state : int
        Random seed

    Yields
    ------
    tuple
        (fold_idx, train_idx, val_idx) for each fold
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    X = df['clean_text'].values
    y = df['category'].values  # Stratify on category

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        yield fold_idx, train_idx, val_idx
