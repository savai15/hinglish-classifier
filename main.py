"""
Hinglish E-Commerce Complaint Classifier - Main Pipeline (Enhanced v2)
=====================================================================
End-to-end pipeline for training, evaluating, and testing
Hinglish complaint classification models.

Enhancements in v2:
- 5-fold cross-validation for reliable evaluation
- Enhanced preprocessing with urgency cue detection
- Hyperparameter tuning with RandomizedSearchCV
- Comprehensive error analysis

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

from src.data_loader import load_data, get_data_stats, split_data, get_cv_splits_multi
from src.preprocessor import HinglishPreprocessor
from src.models import (
    build_tfidf_pipeline,
    build_char_ngram_pipeline,
    build_combined_pipeline,
    build_svm_pipeline,
    get_param_grid,
    get_pipeline_type,
    tune_model,
    train_model,
    predict,
    save_model,
)
from src.evaluation import (
    evaluate_classifier,
    evaluate_cv,
    plot_confusion_matrix,
    plot_normalized_confusion_matrix,
    compare_models,
    plot_per_class_f1,
    plot_cv_comparison,
    print_comparison_table,
    print_cv_table,
)
from src.error_analysis import analyze_errors, compare_error_rates


def main():
    print("=" * 70)
    print("  HINGLISH E-COMMERCE COMPLAINT CLASSIFIER (Enhanced v2)")
    print("  End-to-End Training Pipeline with CV + Tuning + Error Analysis")
    print("=" * 70)

    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("\n[1/8] Loading dataset...")
    csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "hinglish_ecommerce_complaints_360_spelling_variants.csv")
    df = load_data(csv_path)

    stats = get_data_stats(df)
    print(f"\n  Dataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Categories: {stats['num_categories']}")
    print(f"  Urgency levels: {stats['num_urgency_levels']}")
    print(f"  Avg text length: {stats['avg_text_length']:.0f} chars")
    print(f"  Avg word count: {stats['avg_word_count']:.0f} words")

    os.makedirs(os.path.join(PROJECT_ROOT, "data", "processed"), exist_ok=True)
    df.to_csv(os.path.join(PROJECT_ROOT, "data", "processed", "full_dataset.csv"), index=False)

    # ========================================================================
    # STEP 2: Preprocess (Enhanced with urgency cues)
    # ========================================================================
    print("\n[2/8] Preprocessing Hinglish text (Enhanced v2 with urgency cues)...")
    preprocessor = HinglishPreprocessor()

    print("\n  Preprocessing Examples:")
    for i in [0, 50, 100, 200, 300]:
        if i < len(df):
            original = df['text'].iloc[i]
            cleaned = preprocessor.preprocess(original)
            print(f"  Original:  {original[:80]}...")
            print(f"  Cleaned:   {cleaned[:80]}...")
            print()

    df['clean_text'] = df['text'].apply(preprocessor.preprocess)
    df.to_csv(os.path.join(PROJECT_ROOT, "data", "processed", "preprocessed_dataset.csv"), index=False)
    print("  Preprocessing complete.")

    # ========================================================================
    # STEP 3: Split Data (Train/Test)
    # ========================================================================
    print("\n[3/8] Splitting data into train/test...")
    df_train, df_test = split_data(df, test_size=0.2)

    X_train = df_train['clean_text'].values
    y_train_cat = df_train['category'].values
    y_train_urg = df_train['urgency'].values

    X_test = df_test['clean_text'].values
    y_test_cat = df_test['category'].values
    y_test_urg = df_test['urgency'].values

    category_names = sorted(df['category'].unique().tolist())
    urgency_names = sorted(df['urgency'].unique().tolist())

    print(f"  Categories: {category_names}")
    print(f"  Urgency levels: {urgency_names}")

    # ========================================================================
    # STEP 4: 5-Fold Cross-Validation (Baseline Evaluation)
    # ========================================================================
    print("\n[4/8] Running 5-fold Cross-Validation (baseline evaluation)...")
    start_time = time.time()

    cv_models_cat = {
        'TF-IDF + LR': build_tfidf_pipeline(),
        'Char N-gram + LR': build_char_ngram_pipeline(),
        'Combined (Word+Char)': build_combined_pipeline(),
        'TF-IDF + SVM': build_svm_pipeline(),
    }

    cv_models_urg = {
        'TF-IDF + LR': build_tfidf_pipeline(),
        'Char N-gram + LR': build_char_ngram_pipeline(),
        'Combined (Word+Char)': build_combined_pipeline(),
        'TF-IDF + SVM': build_svm_pipeline(),
    }

    print("\n  --- Category Classification CV ---")
    cv_results_cat = {}
    for name, pipeline in cv_models_cat.items():
        cv_results_cat[name] = evaluate_cv(
            pipeline, X_train, y_train_cat,
            cv=5, scoring='f1_macro',
            task_name=f"Category - {name}"
        )

    print("\n  --- Urgency Classification CV ---")
    cv_results_urg = {}
    for name, pipeline in cv_models_urg.items():
        cv_results_urg[name] = evaluate_cv(
            pipeline, X_train, y_train_urg,
            cv=5, scoring='f1_macro',
            task_name=f"Urgency - {name}"
        )

    print_cv_table(cv_results_cat, "Category Classification CV")
    print_cv_table(cv_results_urg, "Urgency Classification CV")

    cv_time = time.time() - start_time
    print(f"  CV completed in {cv_time:.1f}s")

    # ========================================================================
    # STEP 5: Hyperparameter Tuning
    # ========================================================================
    print("\n[5/8] Hyperparameter tuning with RandomizedSearchCV...")
    start_time = time.time()

    # Tune category models
    print("\n  --- Tuning Category Models ---")
    tuned_models_cat = {}
    best_cv_cat = {}

    for name, pipeline in cv_models_cat.items():
        print(f"\n  Tuning {name}...")
        ptype = get_pipeline_type(pipeline)
        param_grid = get_param_grid(ptype)

        best_pipeline, search = tune_model(
            pipeline, param_grid, X_train, y_train_cat,
            n_iter=20, cv=3, scoring='f1_macro', verbose=0
        )

        tuned_models_cat[name] = best_pipeline
        best_cv_cat[name] = {
            'mean': float(search.best_score_),
            'std': 0.0,
            'params': search.best_params_,
        }
        print(f"    Best CV F1: {search.best_score_:.4f}")
        print(f"    Best params: {search.best_params_}")

    # Tune urgency models
    print("\n  --- Tuning Urgency Models ---")
    tuned_models_urg = {}
    best_cv_urg = {}

    for name, pipeline in cv_models_urg.items():
        print(f"\n  Tuning {name}...")
        ptype = get_pipeline_type(pipeline)
        param_grid = get_param_grid(ptype)

        best_pipeline, search = tune_model(
            pipeline, param_grid, X_train, y_train_urg,
            n_iter=20, cv=3, scoring='f1_macro', verbose=0
        )

        tuned_models_urg[name] = best_pipeline
        best_cv_urg[name] = {
            'mean': float(search.best_score_),
            'std': 0.0,
            'params': search.best_params_,
        }
        print(f"    Best CV F1: {search.best_score_:.4f}")
        print(f"    Best params: {search.best_params_}")

    tune_time = time.time() - start_time
    print(f"\n  Tuning completed in {tune_time:.1f}s")

    # ========================================================================
    # STEP 6: Train Final Models and Evaluate on Test Set
    # ========================================================================
    print("\n[6/8] Training final models on full training set and evaluating on test set...")
    start_time = time.time()

    # Category models
    results_cat = {}
    predictions_cat = {}

    print("\n  --- Category Classification (Tuned Models) ---")
    for name, pipeline in tuned_models_cat.items():
        print(f"\n  --- {name} ---")
        t0 = time.time()
        train_model(pipeline, X_train, y_train_cat, verbose=False)
        train_time = time.time() - t0

        y_pred = predict(pipeline, X_test)
        results_cat[name] = evaluate_classifier(
            y_test_cat, y_pred,
            class_names=category_names,
            task_name=f"Category Classification - {name}"
        )
        results_cat[name]['train_time'] = train_time
        results_cat[name]['cv_f1'] = best_cv_cat[name]['mean']
        predictions_cat[name] = y_pred

        model_filename = name.lower().replace(' ', '_').replace('+', '').replace('(', '').replace(')', '')
        save_model(pipeline, os.path.join(PROJECT_ROOT, "models", f"category_{model_filename}.pkl"))

    # Urgency models
    results_urg = {}
    predictions_urg = {}

    print("\n  --- Urgency Classification (Tuned Models) ---")
    for name, pipeline in tuned_models_urg.items():
        print(f"\n  --- {name} ---")
        t0 = time.time()
        train_model(pipeline, X_train, y_train_urg, verbose=False)
        train_time = time.time() - t0

        y_pred = predict(pipeline, X_test)
        results_urg[name] = evaluate_classifier(
            y_test_urg, y_pred,
            class_names=urgency_names,
            task_name=f"Urgency Classification - {name}"
        )
        results_urg[name]['train_time'] = train_time
        results_urg[name]['cv_f1'] = best_cv_urg[name]['mean']
        predictions_urg[name] = y_pred

        model_filename = name.lower().replace(' ', '_').replace('+', '').replace('(', '').replace(')', '')
        save_model(pipeline, os.path.join(PROJECT_ROOT, "models", f"urgency_{model_filename}.pkl"))

    total_time = time.time() - start_time
    print(f"\n  Training + evaluation completed in {total_time:.1f}s")

    # ========================================================================
    # STEP 7: Error Analysis
    # ========================================================================
    print("\n[7/8] Running error analysis...")
    reports_dir = os.path.join(PROJECT_ROOT, "reports")

    # Find best models for error analysis
    best_cat_name = max(results_cat.items(), key=lambda x: x[1]['f1_macro'])[0]
    best_urg_name = max(results_urg.items(), key=lambda x: x[1]['f1_macro'])[0]

    print(f"\n  Best Category model: {best_cat_name}")
    error_results_cat = analyze_errors(
        y_test_cat, predictions_cat[best_cat_name],
        X_test, category_names,
        task_name=f"Category ({best_cat_name})",
        save_dir=reports_dir
    )

    print(f"\n  Best Urgency model: {best_urg_name}")
    error_results_urg = analyze_errors(
        y_test_urg, predictions_urg[best_urg_name],
        X_test, urgency_names,
        task_name=f"Urgency ({best_urg_name})",
        save_dir=reports_dir
    )

    # Compare error rates across all models
    all_error_results_cat = {}
    for name, preds in predictions_cat.items():
        mask = y_test_cat != preds
        all_error_results_cat[name] = {
            'error_rate': float(np.mean(mask)),
            'accuracy': float(np.mean(~mask)),
        }

    all_error_results_urg = {}
    for name, preds in predictions_urg.items():
        mask = y_test_urg != preds
        all_error_results_urg[name] = {
            'error_rate': float(np.mean(mask)),
            'accuracy': float(np.mean(~mask)),
        }

    compare_error_rates(all_error_results_cat, "Category Error Rates", reports_dir)
    compare_error_rates(all_error_results_urg, "Urgency Error Rates", reports_dir)

    # ========================================================================
    # STEP 8: Generate Visualizations and Final Report
    # ========================================================================
    print("\n[8/8] Generating visualizations and final report...")

    # Visualizations
    print_comparison_table(results_cat, "Category Classification (Tuned Models)")
    print_comparison_table(results_urg, "Urgency Classification (Tuned Models)")

    # Confusion matrices for best models
    for name in predictions_cat:
        if name == best_cat_name:
            plot_confusion_matrix(
                y_test_cat, predictions_cat[name],
                class_names=category_names,
                task_name=f"Category CM - {name} (Best)",
                save_path=os.path.join(reports_dir, f"category_cm_{name.lower().replace(' ', '_').replace('+', '_')}.png")
            )
            plot_normalized_confusion_matrix(
                y_test_cat, predictions_cat[name],
                class_names=category_names,
                task_name=f"Category Normalized CM - {name} (Best)",
                save_path=os.path.join(reports_dir, f"category_cm_norm_{name.lower().replace(' ', '_').replace('+', '_')}.png")
            )

    for name in predictions_urg:
        if name == best_urg_name:
            plot_confusion_matrix(
                y_test_urg, predictions_urg[name],
                class_names=urgency_names,
                task_name=f"Urgency CM - {name} (Best)",
                save_path=os.path.join(reports_dir, f"urgency_cm_{name.lower().replace(' ', '_').replace('+', '_')}.png")
            )

    # Model comparison charts
    compare_models(results_cat, "Category Classification", os.path.join(reports_dir, "category_model_comparison.png"))
    compare_models(results_urg, "Urgency Classification", os.path.join(reports_dir, "urgency_model_comparison.png"))

    # Per-class F1
    plot_per_class_f1(results_cat, category_names, "Category Classification",
                      os.path.join(reports_dir, "category_per_class_f1.png"))
    plot_per_class_f1(results_urg, urgency_names, "Urgency Classification",
                      os.path.join(reports_dir, "urgency_per_class_f1.png"))

    # CV comparison charts
    plot_cv_comparison(cv_results_cat, "Category CV", os.path.join(reports_dir, "category_cv_comparison.png"))
    plot_cv_comparison(cv_results_urg, "Urgency CV", os.path.join(reports_dir, "urgency_cv_comparison.png"))

    # Save preprocessor
    preprocessor.save(os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl"))

    # Generate text report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  HINGLISH E-COMMERCE COMPLAINT CLASSIFIER - FINAL REPORT (Enhanced v2)")
    report_lines.append("=" * 70)
    report_lines.append(f"\n  Dataset: {stats['total_samples']} samples")
    report_lines.append(f"  Categories: {stats['num_categories']}")
    report_lines.append(f"  Urgency levels: {stats['num_urgency_levels']}")
    report_lines.append(f"\n  Train: {len(df_train)} | Test: {len(df_test)}")
    report_lines.append(f"  CV Folds: 5")
    report_lines.append(f"  Enhancement: Urgency cues + Hyperparameter tuning + Error analysis")

    report_lines.append(f"\n{'='*70}")
    report_lines.append("  CATEGORY CLASSIFICATION RESULTS (Tuned Models)")
    report_lines.append(f"{'='*70}")

    for name in results_cat:
        r = results_cat[name]
        report_lines.append(f"\n  {name}:")
        report_lines.append(f"    Accuracy:         {r['accuracy']:.4f}")
        report_lines.append(f"    F1 (macro):       {r['f1_macro']:.4f}")
        report_lines.append(f"    F1 (weighted):    {r['f1_weighted']:.4f}")
        report_lines.append(f"    CV F1 (mean):     {r['cv_f1']:.4f}")
        report_lines.append(f"    Train time:       {r['train_time']:.2f}s")

    report_lines.append(f"\n{'='*70}")
    report_lines.append("  URGENCY CLASSIFICATION RESULTS (Tuned Models)")
    report_lines.append(f"{'='*70}")

    for name in results_urg:
        r = results_urg[name]
        report_lines.append(f"\n  {name}:")
        report_lines.append(f"    Accuracy:         {r['accuracy']:.4f}")
        report_lines.append(f"    F1 (macro):       {r['f1_macro']:.4f}")
        report_lines.append(f"    F1 (weighted):    {r['f1_weighted']:.4f}")
        report_lines.append(f"    CV F1 (mean):     {r['cv_f1']:.4f}")
        report_lines.append(f"    Train time:       {r['train_time']:.2f}s")

    best_cat = max(results_cat.items(), key=lambda x: x[1]['f1_macro'])
    best_urg = max(results_urg.items(), key=lambda x: x[1]['f1_macro'])

    report_lines.append(f"\n{'='*70}")
    report_lines.append("  BEST MODELS")
    report_lines.append(f"{'='*70}")
    report_lines.append(f"\n  Best Category Model: {best_cat[0]} (F1={best_cat[1]['f1_macro']:.4f}, CV={best_cat[1]['cv_f1']:.4f})")
    report_lines.append(f"  Best Urgency Model:  {best_urg[0]} (F1={best_urg[1]['f1_macro']:.4f}, CV={best_urg[1]['cv_f1']:.4f})")

    report_lines.append(f"\n{'='*70}")
    report_lines.append("  ENHANCEMENT IMPACT")
    report_lines.append(f"{'='*70}")
    report_lines.append("\n  v2 Enhancements applied:")
    report_lines.append("  1. Urgency cue detection (CAPS, threats, escalation, high amounts)")
    report_lines.append("  2. Repeated letter normalization (urgenttt -> urgent)")
    report_lines.append("  3. 5-fold cross-validation for reliable evaluation")
    report_lines.append("  4. Hyperparameter tuning with RandomizedSearchCV")
    report_lines.append("  5. Comprehensive error analysis")

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
    import numpy as np
    main()
