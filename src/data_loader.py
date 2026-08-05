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


import re
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

def get_template_group(text):
    """
    Mask transaction IDs and numbers to extract the underlying text template.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\b(ORD|TRK|INV|TXN)\d+\b', '[REF]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{4,}\b', '[NUM]', text)
    return " ".join(text.lower().split())


def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into train and test sets grouped by template to prevent leakage.
    """
    df['template_group'] = df['text'].apply(get_template_group)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df['template_group']))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    print(f"  Group-split (Leakage Free): Train: {len(df_train)} samples ({df_train['template_group'].nunique()} templates) | Test: {len(df_test)} samples ({df_test['template_group'].nunique()} templates)")

    return df_train, df_test


def get_cv_splits(df, n_splits=5, random_state=42, label_column='category'):
    """
    Generate GroupKFold cross-validation splits grouped by text template.
    """
    if 'template_group' not in df.columns:
        df['template_group'] = df['text'].apply(get_template_group)

    gkf = GroupKFold(n_splits=n_splits)
    X = df['clean_text'].to_numpy()
    y = df[label_column].to_numpy()
    groups = df['template_group'].to_numpy()

    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        yield train_idx, val_idx


def get_cv_splits_multi(df, n_splits=5, random_state=42):
    """
    Generate GroupKFold CV splits.
    """
    return get_cv_splits(df, n_splits=n_splits, random_state=random_state, label_column='category')

