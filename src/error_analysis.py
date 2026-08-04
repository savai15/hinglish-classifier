"""
Error Analysis Module for Hinglish Complaint Classification

Provides detailed analysis of misclassified samples, confusion patterns,
and failure case identification.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def analyze_errors(y_true, y_pred, X_text, class_names, task_name="Classification",
                   save_dir=None, top_n=10):
    """
    Comprehensive error analysis for a classification task.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    X_text : array-like
        Original text samples
    class_names : list
        List of class names
    task_name : str
        Name of the task for display
    save_dir : str, optional
        Directory to save analysis artifacts
    top_n : int
        Number of top confused pairs to show

    Returns
    -------
    dict
        Error analysis results
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    X_text = np.array(X_text)

    results = {
        'total_samples': len(y_true),
        'total_errors': int(np.sum(y_true != y_pred)),
        'error_rate': float(np.mean(y_true != y_pred)),
        'accuracy': float(np.mean(y_true == y_pred)),
    }

    # Find misclassified indices
    error_mask = y_true != y_pred
    error_indices = np.where(error_mask)[0]

    # Create misclassified samples dataframe
    error_samples = []
    for idx in error_indices:
        error_samples.append({
            'index': int(idx),
            'text': X_text[idx],
            'true_label': y_true[idx],
            'predicted_label': y_pred[idx],
        })

    df_errors = pd.DataFrame(error_samples)
    results['error_samples'] = df_errors

    # Confused pairs analysis
    confused_pairs = Counter()
    for true, pred in zip(y_true[error_mask], y_pred[error_mask]):
        confused_pairs[(true, pred)] += 1

    results['confused_pairs'] = confused_pairs.most_common(top_n)

    # Per-class error rates
    per_class_errors = {}
    for cls in class_names:
        cls_mask = y_true == cls
        cls_total = np.sum(cls_mask)
        cls_errors = np.sum(error_mask & cls_mask)
        per_class_errors[cls] = {
            'total': int(cls_total),
            'errors': int(cls_errors),
            'error_rate': float(cls_errors / cls_total) if cls_total > 0 else 0.0,
            'accuracy': float(1 - cls_errors / cls_total) if cls_total > 0 else 1.0,
        }
    results['per_class_errors'] = per_class_errors

    # Text length analysis
    text_lengths = np.array([len(t.split()) for t in X_text])
    short_mask = text_lengths <= 6
    medium_mask = (text_lengths > 6) & (text_lengths <= 12)
    long_mask = text_lengths > 12

    results['length_analysis'] = {
        'short_le6': {
            'count': int(np.sum(short_mask)),
            'error_rate': float(np.mean(error_mask & short_mask)) if np.sum(short_mask) > 0 else 0.0,
        },
        'medium_7_12': {
            'count': int(np.sum(medium_mask)),
            'error_rate': float(np.mean(error_mask & medium_mask)) if np.sum(medium_mask) > 0 else 0.0,
        },
        'long_gt12': {
            'count': int(np.sum(long_mask)),
            'error_rate': float(np.mean(error_mask & long_mask)) if np.sum(long_mask) > 0 else 0.0,
        },
    }

    # Print analysis
    print(f"\n{'='*60}")
    print(f"  ERROR ANALYSIS: {task_name}")
    print(f"{'='*60}")
    print(f"\n  Total samples: {results['total_samples']}")
    print(f"  Misclassified: {results['total_errors']}")
    print(f"  Error rate: {results['error_rate']:.1%}")
    print(f"  Accuracy: {results['accuracy']:.4f}")

    print(f"\n  --- Most Confused Class Pairs ---")
    for (true_cls, pred_cls), count in results['confused_pairs']:
        print(f"    {true_cls} -> {pred_cls}: {count} cases")

    print(f"\n  --- Per-Class Error Rates ---")
    for cls in sorted(per_class_errors.keys()):
        info = per_class_errors[cls]
        print(f"    {cls:<30} {info['errors']}/{info['total']} errors ({info['error_rate']:.1%})")

    print(f"\n  --- Text Length Analysis ---")
    for length_bin, info in results['length_analysis'].items():
        print(f"    {length_bin:<20} {info['count']} samples, error rate: {info['error_rate']:.1%}")

    print(f"\n  --- Sample Misclassifications ---")
    for _, row in df_errors.head(5).iterrows():
        text_short = row['text'][:60] + "..." if len(row['text']) > 60 else row['text']
        print(f"    True: {row['true_label']:<20} Pred: {row['predicted_label']:<20} | {text_short}")

    print(f"{'='*60}\n")

    # Save error report if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        _save_error_report(results, class_names, task_name, save_dir)
        _plot_confused_pairs(results, task_name, save_dir)
        _plot_per_class_accuracy(results, class_names, task_name, save_dir)

    return results


def _save_error_report(results, class_names, task_name, save_dir):
    """Save a human-readable error report to a text file."""
    report_path = os.path.join(save_dir, f"{task_name.lower().replace(' ', '_')}_error_analysis.txt")

    lines = []
    lines.append("=" * 60)
    lines.append(f"  ERROR ANALYSIS REPORT: {task_name}")
    lines.append("=" * 60)
    lines.append(f"\n  Total samples: {results['total_samples']}")
    lines.append(f"  Misclassified: {results['total_errors']}")
    lines.append(f"  Error rate: {results['error_rate']:.1%}")
    lines.append(f"  Accuracy: {results['accuracy']:.4f}")

    lines.append(f"\n{'='*60}")
    lines.append("  MOST CONFUSED CLASS PAIRS")
    lines.append(f"{'='*60}")
    for (true_cls, pred_cls), count in results['confused_pairs']:
        lines.append(f"  {true_cls} -> {pred_cls}: {count} cases")

    lines.append(f"\n{'='*60}")
    lines.append("  PER-CLASS ERROR RATES")
    lines.append(f"{'='*60}")
    for cls in sorted(results['per_class_errors'].keys()):
        info = results['per_class_errors'][cls]
        lines.append(f"  {cls:<30} {info['errors']}/{info['total']} errors ({info['error_rate']:.1%})")

    lines.append(f"\n{'='*60}")
    lines.append("  TEXT LENGTH ANALYSIS")
    lines.append(f"{'='*60}")
    for length_bin, info in results['length_analysis'].items():
        lines.append(f"  {length_bin:<20} {info['count']} samples, error rate: {info['error_rate']:.1%}")

    lines.append(f"\n{'='*60}")
    lines.append("  ALL MISCLASSIFIED SAMPLES")
    lines.append(f"{'='*60}")
    for _, row in results['error_samples'].iterrows():
        lines.append(f"\n  [{row['index']}]")
        lines.append(f"    Text: {row['text']}")
        lines.append(f"    True: {row['true_label']}")
        lines.append(f"    Pred: {row['predicted_label']}")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Error report saved to {report_path}")


def _plot_confused_pairs(results, task_name, save_dir):
    """Plot top confused class pairs as a bar chart."""
    if not results['confused_pairs']:
        return

    pairs = [f"{t} -> {p}" for (t, p), _ in results['confused_pairs']]
    counts = [c for _, c in results['confused_pairs']]

    fig, ax = plt.subplots(figsize=(10, max(4, len(pairs) * 0.5)))
    bars = ax.barh(pairs, counts, color='#F44336', alpha=0.8)
    ax.set_xlabel('Number of Cases', fontsize=12)
    ax.set_title(f'{task_name} - Most Confused Class Pairs', fontsize=14)
    ax.invert_yaxis()

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{task_name.lower().replace(' ', '_')}_confused_pairs.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confused pairs chart saved to {save_path}")


def _plot_per_class_accuracy(results, class_names, task_name, save_dir):
    """Plot per-class accuracy as a bar chart."""
    classes = sorted(results['per_class_errors'].keys())
    accuracies = [results['per_class_errors'][c]['accuracy'] for c in classes]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#4CAF50' if acc >= 0.7 else '#FF9800' if acc >= 0.5 else '#F44336' for acc in accuracies]
    bars = ax.bar(classes, accuracies, color=colors, alpha=0.8)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'{task_name} - Per-Class Accuracy', fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=np.mean(accuracies), color='black', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(accuracies):.3f}')
    ax.legend(fontsize=10)
    plt.xticks(rotation=45, ha='right')

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.2f}', ha='center', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{task_name.lower().replace(' ', '_')}_per_class_accuracy.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Per-class accuracy chart saved to {save_path}")


def compare_error_rates(error_results_dict, task_name="Error Rate Comparison", save_dir=None):
    """
    Compare error rates across multiple models.

    Parameters
    ----------
    error_results_dict : dict
        Dictionary mapping model names to error analysis results
    task_name : str
        Task name for display
    save_dir : str, optional
        Directory to save comparison chart
    """
    models = list(error_results_dict.keys())
    error_rates = [error_results_dict[m]['error_rate'] for m in models]
    accuracies = [error_results_dict[m]['accuracy'] for m in models]

    print(f"\n{'='*60}")
    print(f"  {task_name} - Error Rate Comparison")
    print(f"{'='*60}")
    print(f"  {'Model':<25} {'Accuracy':>10} {'Error Rate':>12}")
    print(f"  {'-'*47}")
    for model, acc, err in zip(models, accuracies, error_rates):
        print(f"  {model:<25} {acc:>10.4f} {err:>11.1%}")
    print(f"{'='*60}\n")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(models))
        bars = ax.bar(x, error_rates, color='#F44336', alpha=0.8)
        ax.set_ylabel('Error Rate', fontsize=12)
        ax.set_title(f'{task_name}', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylim(0, max(error_rates) * 1.3 if error_rates else 1)

        for bar, err in zip(bars, error_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{err:.1%}', ha='center', fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{task_name.lower().replace(' ', '_')}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Error comparison chart saved to {save_path}")
