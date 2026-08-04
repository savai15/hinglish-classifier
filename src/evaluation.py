"""
Evaluation Module for Hinglish Complaint Classification

Provides comprehensive evaluation metrics, confusion matrices,
and comparison visualizations.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def evaluate_classifier(y_true, y_pred, class_names=None, task_name="Classification"):
    """
    Comprehensive evaluation of a classifier.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list, optional
        List of class names for display
    task_name : str
        Name of the task (for display)

    Returns
    -------
    dict
        Dictionary with all evaluation metrics
    """
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    results['per_class'] = report

    print(f"\n{'='*60}")
    print(f"  {task_name} Results")
    print(f"{'='*60}")
    print(f"  Accuracy:           {results['accuracy']:.4f}")
    print(f"  Precision (macro):  {results['precision_macro']:.4f}")
    print(f"  Recall (macro):     {results['recall_macro']:.4f}")
    print(f"  F1-score (macro):   {results['f1_macro']:.4f}")
    print(f"  F1-score (weighted):{results['f1_weighted']:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print(f"{'='*60}\n")

    return results


def plot_confusion_matrix(y_true, y_pred, class_names=None, task_name="Confusion Matrix",
                          save_path=None, figsize=(10, 8)):
    """
    Plot and optionally save a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(task_name, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Confusion matrix saved to {save_path}")

    plt.close()
    return cm


def plot_normalized_confusion_matrix(y_true, y_pred, class_names=None,
                                     task_name="Normalized Confusion Matrix",
                                     save_path=None, figsize=(10, 8)):
    """
    Plot a normalized (percentage) confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_names, normalize='true')
    cm_pct = cm * 100

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_pct, annot=True, fmt='.1f', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0, vmax=100
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(task_name, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Normalized confusion matrix saved to {save_path}")

    plt.close()


def compare_models(results_dict, task_name="Model Comparison", save_path=None, figsize=(12, 6)):
    """
    Compare multiple models using bar charts.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping model names to their evaluation results
    """
    models = list(results_dict.keys())
    metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    metric_labels = ['Accuracy', 'F1 (macro)', 'Precision (macro)', 'Recall (macro)']

    data = []
    for model_name in models:
        for metric, label in zip(metrics, metric_labels):
            data.append({
                'Model': model_name,
                'Metric': label,
                'Score': results_dict[model_name][metric]
            })

    df_compare = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Bar plot
    ax = axes[0]
    x = np.arange(len(models))
    width = 0.2
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        scores = [results_dict[m][metric] for m in models]
        ax.bar(x + i * width, scores, width, label=label, color=colors[i])

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{task_name} - Score Comparison', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # Summary table
    ax2 = axes[1]
    ax2.axis('off')
    table_data = [['Model'] + metric_labels]
    for model_name in models:
        row = [model_name]
        for metric in metrics:
            row.append(f"{results_dict[model_name][metric]:.4f}")
        table_data.append(row)

    table = ax2.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax2.set_title(f'{task_name} - Detailed Scores', fontsize=14, pad=20)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Comparison chart saved to {save_path}")

    plt.close()
    return df_compare


def plot_per_class_f1(results_dict, class_names, task_name="Per-Class F1 Scores",
                      save_path=None, figsize=(14, 6)):
    """
    Plot per-class F1 scores for multiple models.
    """
    data = []
    for model_name, results in results_dict.items():
        for cls in class_names:
            if cls in results.get('per_class', {}):
                f1 = results['per_class'][cls]['f1-score']
                data.append({'Model': model_name, 'Class': cls, 'F1-Score': f1})

    df_f1 = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=df_f1, x='Class', y='F1-Score', hue='Model', ax=ax)
    ax.set_title(f'{task_name} - Per-Class F1 Scores', fontsize=14)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right')
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Per-class F1 chart saved to {save_path}")

    plt.close()


def print_comparison_table(results_dict, task_name="Model Comparison"):
    """
    Print a formatted comparison table of model results.
    """
    models = list(results_dict.keys())
    metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
    labels = ['Accuracy', 'F1 (macro)', 'F1 (weighted)', 'Precision (macro)', 'Recall (macro)']

    header = f"\n{'='*70}\n  {task_name}\n{'='*70}"
    print(header)
    print(f"  {'Model':<25} {'Accuracy':>10} {'F1-macro':>10} {'F1-wtd':>10} {'Prec-macro':>12} {'Rec-macro':>10}")
    print(f"  {'-'*65}")

    for model_name in models:
        row = f"  {model_name:<25}"
        for metric in metrics:
            val = results_dict[model_name].get(metric, 0)
            row += f" {val:>10.4f}"
        print(row)

    print(f"{'='*70}\n")
