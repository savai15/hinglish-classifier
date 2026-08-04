"""
Data Loader for Hinglish E-Commerce Complaints
Loads, validates, and splits the complaint dataset.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split


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


def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train, validation, and test sets with stratification.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    test_size : float
        Fraction for test set
    val_size : float
        Fraction for validation set (taken from train after test split)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        (df_train, df_val, df_test) - three DataFrames
    """
    # First split: train+val vs test
    df_trainval, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['category']
    )

    # Second split: train vs val (adjust val_size relative to trainval)
    adjusted_val_size = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=df_trainval['category']
    )

    print(f"  Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)
