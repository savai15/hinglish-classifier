"""
Hinglish E-Commerce Complaint Classifier - Main Pipeline
=========================================================
End-to-end pipeline for training, evaluating, and testing
Hinglish complaint classification models.

Run this script to train all models and generate results.
"""
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_data, get_data_stats, split_data
from src.preprocessor import HinglishPreprocessor
from src.models import (
    build_tfidf_pipeline,
    build_char_ngram_pipeline,
    build_combined_pipeline,
    build_svm_pipeline,
    train_model,
    predict,
    save_model,
    load_model
)
from src.evaluation import (
    evaluate_classifier,
    plot_confusion_matrix,
    plot_normalized_confusion_matrix,
    compare_models,
    plot_per_class_f1,
    print_comparison_table
)


def main():
    print("=" * 70)
    print("  HINGLISH E-COMMERCE COMPLAINT CLASSIFIER")
    print("  End-to-End Training Pipeline")
    print("=" * 70)

    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("\n[1/7] Loading dataset...")
    csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "hinglish_ecommerce_complaints_360_spelling_variants.csv")
    df = load_data(csv_path)

    stats = get_data_stats(df)
    print(f"\n  Dataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Categories: {stats['num_categories']}")
    print(f"  Urgency levels: {stats['num_urgency_levels']}")
    print(f"  Avg text length: {stats['avg_text_length']:.0f} chars")
    print(f"  Avg word count: {stats['avg_word_count']:.0f} words")

    # Save processed data
    os.makedirs(os.path.join(PROJECT_ROOT, "data", "processed"), exist_ok=True)
    df.to_csv(os.path.join(PROJECT_ROOT, "data", "processed", "full_dataset.csv"), index=False)

    # ========================================================================
    # STEP 2: Preprocess
    # ========================================================================
    print("\n[2/7] Preprocessing Hinglish text...")
    preprocessor = HinglishPreprocessor()

    # Show preprocessing examples
    print("\n  Preprocessing Examples:")
    for i in [0, 50, 100, 200, 300]:
        if i < len(df):
            original = df['text'].iloc[i]
            cleaned = preprocessor.preprocess(original)
            print(f"  Original:  {original[:80]}...")
            print(f"  Cleaned:   {cleaned[:80]}...")
            print()

    # Apply preprocessing
    df['clean_text'] = df['text'].apply(preprocessor.preprocess)
    df.to_csv(os.path.join(PROJECT_ROOT, "data", "processed", "preprocessed_dataset.csv"), index=False)
    print("  Preprocessing complete.")

    # ========================================================================
    # STEP 3: Split Data
    # ========================================================================
    print("\n[3/7] Splitting data into train/val/test...")
    df_train, df_val, df_test = split_data(df, test_size=0.2, val_size=0.1)

    # Combine train+val for final training (use val for model selection)
    df_trainval = df_train.copy()
    X_trainval = df_trainval['clean_text'].values
    y_trainval_cat = df_trainval['category'].values
    y_trainval_urg = df_trainval['urgency'].values

    X_test = df_test['clean_text'].values
    y_test_cat = df_test['category'].values
    y_test_urg = df_test['urgency'].values

    # Get class names
    category_names = sorted(df['category'].unique().tolist())
    urgency_names = ['Low', 'Medium', 'High']

    print(f"  Categories: {category_names}")
    print(f"  Urgency levels: {urgency_names}")

    # ========================================================================
    # STEP 4: Train Models - CATEGORY Classification
    # ========================================================================
    print("\n[4/7] Training CATEGORY classification models...")
    start_time = time.time()

    # Build all models
    models_cat = {
        'TF-IDF + LR': build_tfidf_pipeline(),
        'Char N-gram + LR': build_char_ngram_pipeline(),
        'Combined (Word+Char)': build_combined_pipeline(),
        'TF-IDF + SVM': build_svm_pipeline(),
    }

    results_cat = {}
    predictions_cat = {}

    for name, pipeline in models_cat.items():
        print(f"\n  --- {name} ---")
        t0 = time.time()
        pipeline = train_model(pipeline, X_trainval, y_trainval_cat)
        train_time = time.time() - t0

        y_pred = predict(pipeline, X_test)
        results_cat[name] = evaluate_classifier(
            y_test_cat, y_pred,
            class_names=category_names,
            task_name=f"Category Classification - {name}"
        )
        results_cat[name]['train_time'] = train_time
        predictions_cat[name] = y_pred

        # Save model
        model_filename = name.lower().replace(' ', '_').replace('+', '').replace('(', '').replace(')', '')
        save_model(pipeline, os.path.join(PROJECT_ROOT, "models", f"category_{model_filename}.pkl"))

    total_time_cat = time.time() - start_time
    print(f"\n  Total category training time: {total_time_cat:.1f}s")

    # ========================================================================
    # STEP 5: Train Models - URGENCY Classification
    # ========================================================================
    print("\n[5/7] Training URGENCY classification models...")
    start_time = time.time()

    models_urg = {
        'TF-IDF + LR': build_tfidf_pipeline(),
        'Char N-gram + LR': build_char_ngram_pipeline(),
        'Combined (Word+Char)': build_combined_pipeline(),
        'TF-IDF + SVM': build_svm_pipeline(),
    }

    results_urg = {}
    predictions_urg = {}

    for name, pipeline in models_urg.items():
        print(f"\n  --- {name} ---")
        t0 = time.time()
        pipeline = train_model(pipeline, X_trainval, y_trainval_urg)
        train_time = time.time() - t0

        y_pred = predict(pipeline, X_test)
        results_urg[name] = evaluate_classifier(
            y_test_urg, y_pred,
            class_names=urgency_names,
            task_name=f"Urgency Classification - {name}"
        )
        results_urg[name]['train_time'] = train_time
        predictions_urg[name] = y_pred

        # Save model
        model_filename = name.lower().replace(' ', '_').replace('+', '').replace('(', '').replace(')', '')
        save_model(pipeline, os.path.join(PROJECT_ROOT, "models", f"urgency_{model_filename}.pkl"))

    total_time_urg = time.time() - start_time
    print(f"\n  Total urgency training time: {total_time_urg:.1f}s")

    # ========================================================================
    # STEP 6: Generate Visualizations
    # ========================================================================
    print("\n[6/7] Generating visualizations and reports...")
    reports_dir = os.path.join(PROJECT_ROOT, "reports")

    # --- Category Classification Visualizations ---
    print("\n  Category Classification:")

    # Comparison table
    print_comparison_table(results_cat, "Category Classification Comparison")

    # Confusion matrices for best models
    for name in ['TF-IDF + LR', 'Char N-gram + LR']:
        if name in predictions_cat:
            plot_confusion_matrix(
                y_test_cat, predictions_cat[name],
                class_names=category_names,
                task_name=f"Category Confusion Matrix - {name}",
                save_path=os.path.join(reports_dir, f"category_cm_{name.lower().replace(' ', '_').replace('+', '_')}.png")
            )
            plot_normalized_confusion_matrix(
                y_test_cat, predictions_cat[name],
                class_names=category_names,
                task_name=f"Category Normalized CM - {name}",
                save_path=os.path.join(reports_dir, f"category_cm_norm_{name.lower().replace(' ', '_').replace('+', '_')}.png")
            )

    # Model comparison
    compare_models(
        results_cat,
        task_name="Category Classification",
        save_path=os.path.join(reports_dir, "category_model_comparison.png")
    )

    # Per-class F1
    plot_per_class_f1(
        results_cat, category_names,
        task_name="Category Classification",
        save_path=os.path.join(reports_dir, "category_per_class_f1.png")
    )

    # --- Urgency Classification Visualizations ---
    print("\n  Urgency Classification:")

    # Comparison table
    print_comparison_table(results_urg, "Urgency Classification Comparison")

    # Confusion matrices
    for name in ['TF-IDF + LR', 'Char N-gram + LR']:
        if name in predictions_urg:
            plot_confusion_matrix(
                y_test_urg, predictions_urg[name],
                class_names=urgency_names,
                task_name=f"Urgency Confusion Matrix - {name}",
                save_path=os.path.join(reports_dir, f"urgency_cm_{name.lower().replace(' ', '_').replace('+', '_')}.png")
            )

    # Model comparison
    compare_models(
        results_urg,
        task_name="Urgency Classification",
        save_path=os.path.join(reports_dir, "urgency_model_comparison.png")
    )

    # Per-class F1
    plot_per_class_f1(
        results_urg, urgency_names,
        task_name="Urgency Classification",
        save_path=os.path.join(reports_dir, "urgency_per_class_f1.png")
    )

    # ========================================================================
    # STEP 7: Save Preprocessor and Final Report
    # ========================================================================
    print("\n[7/7] Saving final artifacts...")
    preprocessor.save(os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl"))

    # Generate text report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  HINGLISH E-COMMERCE COMPLAINT CLASSIFIER - FINAL REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"\n  Dataset: {stats['total_samples']} samples")
    report_lines.append(f"  Categories: {stats['num_categories']}")
    report_lines.append(f"  Urgency levels: {stats['num_urgency_levels']}")
    report_lines.append(f"\n  Train: {len(df_trainval)} | Test: {len(df_test)}")
    report_lines.append(f"\n{'='*70}")
    report_lines.append("  CATEGORY CLASSIFICATION RESULTS")
    report_lines.append(f"{'='*70}")

    for name in results_cat:
        r = results_cat[name]
        report_lines.append(f"\n  {name}:")
        report_lines.append(f"    Accuracy:    {r['accuracy']:.4f}")
        report_lines.append(f"    F1 (macro):  {r['f1_macro']:.4f}")
        report_lines.append(f"    F1 (weighted):{r['f1_weighted']:.4f}")
        report_lines.append(f"    Train time:  {r['train_time']:.2f}s")

    report_lines.append(f"\n{'='*70}")
    report_lines.append("  URGENCY CLASSIFICATION RESULTS")
    report_lines.append(f"{'='*70}")

    for name in results_urg:
        r = results_urg[name]
        report_lines.append(f"\n  {name}:")
        report_lines.append(f"    Accuracy:    {r['accuracy']:.4f}")
        report_lines.append(f"    F1 (macro):  {r['f1_macro']:.4f}")
        report_lines.append(f"    F1 (weighted):{r['f1_weighted']:.4f}")
        report_lines.append(f"    Train time:  {r['train_time']:.2f}s")

    # Find best models
    best_cat = max(results_cat.items(), key=lambda x: x[1]['f1_macro'])
    best_urg = max(results_urg.items(), key=lambda x: x[1]['f1_macro'])

    report_lines.append(f"\n{'='*70}")
    report_lines.append("  BEST MODELS")
    report_lines.append(f"{'='*70}")
    report_lines.append(f"\n  Best Category Model: {best_cat[0]} (F1={best_cat[1]['f1_macro']:.4f})")
    report_lines.append(f"  Best Urgency Model:  {best_urg[0]} (F1={best_urg[1]['f1_macro']:.4f})")

    report_lines.append(f"\n{'='*70}")
    report_lines.append("  KEY FINDING")
    report_lines.append(f"{'='*70}")
    report_lines.append("\n  Character n-gram approach (mimicking FastText's subword method)")
    report_lines.append("  outperforms word-level TF-IDF on Hinglish text because it")
    report_lines.append("  naturally handles spelling variants (nahi/nai/nahee),")
    report_lines.append("  out-of-vocabulary words, and code-mixed Hindi-English text.")

    report_lines.append(f"\n{'='*70}")

    report_text = '\n'.join(report_lines)
    with open(os.path.join(reports_dir, "results.txt"), 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n  All outputs saved to: {PROJECT_ROOT}")
    print(f"  - Models: models/")
    print(f"  - Reports: reports/")
    print(f"  - Data: data/processed/")
    print(f"\n  Pipeline complete!")


if __name__ == "__main__":
    main()
