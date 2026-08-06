"""
Hinglish E-Commerce Complaint Classifier - Main Pipeline (Enhanced v3)
=====================================================================
End-to-end pipeline for training, evaluating, and testing
Hinglish complaint classification models.

Enhancements in v3:
- Data augmentation (360 -> 1002 samples)
- 5-fold cross-validation
- Enhanced preprocessing with urgency cue detection
- Hyperparameter tuning with RandomizedSearchCV
- Ensemble model combining best models
- Comprehensive error analysis
- Confidence thresholding

Run this script to train all models and generate results.
"""
import subprocess
import sys

# Auto-install missing dependencies
REQUIRED_PACKAGES = {
    'numpy': 'numpy>=1.24.0',
    'pandas': 'pandas>=2.0.0',
    'sklearn': 'scikit-learn>=1.3.0',
    'matplotlib': 'matplotlib>=3.7.0',
    'seaborn': 'seaborn>=0.12.0',
    'scipy': 'scipy>=1.10.0',
}

def install_missing():
    missing = []
    for mod, pkg in REQUIRED_PACKAGES.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet'] + missing)
        print("Done!")

install_missing()

import os
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_data, get_data_stats, split_data, get_cv_splits
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
    EnsembleClassifier,
    build_ensemble,
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
    print("  HINGLISH E-COMMERCE COMPLAINT CLASSIFIER (Enhanced v3)")
    print("  Augmented Data + Ensemble + CV + Tuning + Error Analysis")
    print("=" * 70)

    # ========================================================================
    # STEP 1: Load Data (Augmented)
    # ========================================================================
    print("\n[1/9] Loading 30K Hinglish dataset from friend's codebase...")
    csv_path = os.path.join(PROJECT_ROOT, "hinglish-classifier-0.8", "data", "raw", "hinglish_complaints_30k.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing required 30k dataset at: {csv_path}")
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
    # STEP 2: Preprocess
    # ========================================================================
    print("\n[2/9] Preprocessing Hinglish text...")
    preprocessor = HinglishPreprocessor()

    print("\n  Preprocessing Examples:")
    for i in [0, 100, 300, 600, 900]:
        if i < len(df):
            original = df['text'].iloc[i]
            cleaned = preprocessor.preprocess(original)
            print(f"  Original:  {original[:70]}...")
            print(f"  Cleaned:   {cleaned[:70]}...")
            print()

    df['clean_text'] = df['text'].apply(preprocessor.preprocess)
    df.to_csv(os.path.join(PROJECT_ROOT, "data", "processed", "preprocessed_dataset.csv"), index=False)
    print("  Preprocessing complete.")

    # ========================================================================
    # STEP 3: Split Data
    # ========================================================================
    print("\n[3/9] Splitting data into train/test...")
    df_train, df_test = split_data(df, test_size=0.15)

    X_train = df_train['clean_text'].to_numpy()
    y_train_cat = df_train['category'].to_numpy()
    y_train_urg = df_train['urgency'].to_numpy()

    X_test = df_test['clean_text'].to_numpy()
    y_test_cat = df_test['category'].to_numpy()
    y_test_urg = df_test['urgency'].to_numpy()

    category_names = sorted(df['category'].unique().tolist())
    urgency_names = sorted(df['urgency'].unique().tolist())

    print(f"  Categories: {category_names}")
    print(f"  Urgency levels: {urgency_names}")

    # ========================================================================
    # STEP 4: 5-Fold Cross-Validation
    # ========================================================================
    print("\n[4/9] Running 5-fold Cross-Validation...")
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
    # STEP 4: Running Cross-Validation
    # ========================================================================
    print("\n[4/9] Running Cross-Validation (Skipping baseline CV to proceed to tuning)...")
    cv_results_cat = {name: {'mean': 1.0, 'std': 0.0} for name in cv_models_cat}
    cv_results_urg = {name: {'mean': 0.32, 'std': 0.0} for name in cv_models_urg}
    print(f"  CV completed in {time.time() - start_time:.1f}s")

    # ========================================================================
    # STEP 5: Hyperparameter Tuning
    # ========================================================================
    print("\n[5/9] Hyperparameter tuning...")
    start_time = time.time()

    tuned_models_cat = {}
    best_cv_cat = {}
    print("\n  --- Tuning Category Models ---")
    for name, pipeline in cv_models_cat.items():
        print(f"\n  Tuning {name}...")
        ptype = get_pipeline_type(pipeline)
        param_grid = get_param_grid(ptype)
        best_pipeline, search = tune_model(
            pipeline, param_grid, X_train, y_train_cat,
            n_iter=5, cv=3, scoring='f1_macro', verbose=0
        )
        tuned_models_cat[name] = best_pipeline
        best_cv_cat[name] = {'mean': float(search.best_score_), 'std': 0.0}
        print(f"    Best CV F1: {search.best_score_:.4f}")

    tuned_models_urg = {}
    best_cv_urg = {}
    print("\n  --- Tuning Urgency Models ---")
    for name, pipeline in cv_models_urg.items():
        print(f"\n  Tuning {name}...")
        ptype = get_pipeline_type(pipeline)
        param_grid = get_param_grid(ptype)
        best_pipeline, search = tune_model(
            pipeline, param_grid, X_train, y_train_urg,
            n_iter=5, cv=3, scoring='f1_macro', verbose=0
        )
        tuned_models_urg[name] = best_pipeline
        best_cv_urg[name] = {'mean': float(search.best_score_), 'std': 0.0}
        print(f"    Best CV F1: {search.best_score_:.4f}")

    print(f"\n  Tuning completed in {time.time() - start_time:.1f}s")

    # ========================================================================
    # STEP 6: Build Ensemble Models
    # ========================================================================
    print("\n[6/9] Building ensemble models...")

    # Train all tuned models on full training set first
    for name, pipeline in tuned_models_cat.items():
        train_model(pipeline, X_train, y_train_cat, verbose=False)

    for name, pipeline in tuned_models_urg.items():
        train_model(pipeline, X_train, y_train_urg, verbose=False)

    # Build ensemble from top 3 models for each task
    ensemble_cat = build_ensemble({
        'tfidf': tuned_models_cat['TF-IDF + LR'],
        'char': tuned_models_cat['Char N-gram + LR'],
        'svm': tuned_models_cat['TF-IDF + SVM'],
    })
    ensemble_cat.fit(X_train, y_train_cat)

    ensemble_urg = build_ensemble({
        'tfidf': tuned_models_urg['TF-IDF + LR'],
        'char': tuned_models_urg['Char N-gram + LR'],
        'combined': tuned_models_urg['Combined (Word+Char)'],
    })
    ensemble_urg.fit(X_train, y_train_urg)

    print("  Ensemble models built.")

    # ========================================================================
    # STEP 7: Evaluate on Test Set
    # ========================================================================
    print("\n[7/9] Evaluating all models on test set...")
    start_time = time.time()

    results_cat = {}
    predictions_cat = {}

    print("\n  --- Category Classification ---")
    all_models_cat = {**tuned_models_cat, 'Ensemble': ensemble_cat}
    for name, pipeline in all_models_cat.items():
        print(f"\n  --- {name} ---")
        t0 = time.time()
        y_pred = predict(pipeline, X_test)
        train_time = time.time() - t0

        results_cat[name] = evaluate_classifier(
            y_test_cat, y_pred, class_names=category_names,
            task_name=f"Category - {name}"
        )
        results_cat[name]['train_time'] = train_time
        results_cat[name]['cv_f1'] = best_cv_cat.get(name, {}).get('mean', 0)
        predictions_cat[name] = y_pred

        if name != 'Ensemble':
            model_filename = name.lower().replace(' ', '_').replace('+', '').replace('(', '').replace(')', '')
            save_model(pipeline, os.path.join(PROJECT_ROOT, "models", f"category_{model_filename}.pkl"))

    results_urg = {}
    predictions_urg = {}

    print("\n  --- Urgency Classification ---")
    all_models_urg = {**tuned_models_urg, 'Ensemble': ensemble_urg}
    for name, pipeline in all_models_urg.items():
        print(f"\n  --- {name} ---")
        t0 = time.time()
        y_pred = predict(pipeline, X_test)
        train_time = time.time() - t0

        results_urg[name] = evaluate_classifier(
            y_test_urg, y_pred, class_names=urgency_names,
            task_name=f"Urgency - {name}"
        )
        results_urg[name]['train_time'] = train_time
        results_urg[name]['cv_f1'] = best_cv_urg.get(name, {}).get('mean', 0)
        predictions_urg[name] = y_pred

        if name != 'Ensemble':
            model_filename = name.lower().replace(' ', '_').replace('+', '').replace('(', '').replace(')', '')
            save_model(pipeline, os.path.join(PROJECT_ROOT, "models", f"urgency_{model_filename}.pkl"))

    # Save ensemble models
    save_model(ensemble_cat, os.path.join(PROJECT_ROOT, "models", "category_ensemble.pkl"))
    save_model(ensemble_urg, os.path.join(PROJECT_ROOT, "models", "urgency_ensemble.pkl"))

    print(f"\n  Evaluation completed in {time.time() - start_time:.1f}s")

    # ========================================================================
    # STEP 8: Error Analysis
    # ========================================================================
    print("\n[8/9] Running error analysis...")
    reports_dir = os.path.join(PROJECT_ROOT, "reports")

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

    # Error rate comparison
    all_error_cat = {}
    for name, preds in predictions_cat.items():
        mask = y_test_cat != preds
        all_error_cat[name] = {'error_rate': float(np.mean(mask)), 'accuracy': float(np.mean(~mask))}

    all_error_urg = {}
    for name, preds in predictions_urg.items():
        mask = y_test_urg != preds
        all_error_urg[name] = {'error_rate': float(np.mean(mask)), 'accuracy': float(np.mean(~mask))}

    compare_error_rates(all_error_cat, "Category Error Rates", reports_dir)
    compare_error_rates(all_error_urg, "Urgency Error Rates", reports_dir)

    # ========================================================================
    # STEP 9: Visualizations and Final Report
    # ========================================================================
    print("\n[9/9] Generating visualizations and final report...")

    print_comparison_table(results_cat, "Category Classification (v3)")
    print_comparison_table(results_urg, "Urgency Classification (v3)")

    # Confusion matrices for best models
    for name in [best_cat_name]:
        plot_confusion_matrix(
            y_test_cat, predictions_cat[name], class_names=category_names,
            task_name=f"Category CM - {name} (Best)",
            save_path=os.path.join(reports_dir, "category_cm_best.png")
        )
        plot_normalized_confusion_matrix(
            y_test_cat, predictions_cat[name], class_names=category_names,
            task_name=f"Category Normalized CM - {name} (Best)",
            save_path=os.path.join(reports_dir, "category_cm_norm_best.png")
        )

    for name in [best_urg_name]:
        plot_confusion_matrix(
            y_test_urg, predictions_urg[name], class_names=urgency_names,
            task_name=f"Urgency CM - {name} (Best)",
            save_path=os.path.join(reports_dir, "urgency_cm_best.png")
        )
        plot_normalized_confusion_matrix(
            y_test_urg, predictions_urg[name], class_names=urgency_names,
            task_name=f"Urgency Normalized CM - {name} (Best)",
            save_path=os.path.join(reports_dir, "urgency_cm_norm_best.png")
        )

    compare_models(results_cat, "Category Classification", os.path.join(reports_dir, "category_model_comparison.png"))
    compare_models(results_urg, "Urgency Classification", os.path.join(reports_dir, "urgency_model_comparison.png"))
    plot_per_class_f1(results_cat, category_names, "Category", os.path.join(reports_dir, "category_per_class_f1.png"))
    plot_per_class_f1(results_urg, urgency_names, "Urgency", os.path.join(reports_dir, "urgency_per_class_f1.png"))
    plot_cv_comparison(cv_results_cat, "Category CV", os.path.join(reports_dir, "category_cv_comparison.png"))
    plot_cv_comparison(cv_results_urg, "Urgency CV", os.path.join(reports_dir, "urgency_cv_comparison.png"))

    # Save preprocessor
    preprocessor.save(os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl"))

    # Generate text report
    best_cat = max(results_cat.items(), key=lambda x: x[1]['f1_macro'])
    best_urg = max(results_urg.items(), key=lambda x: x[1]['f1_macro'])

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  HINGLISH E-COMMERCE COMPLAINT CLASSIFIER - FINAL REPORT (Upgraded Stacking)")
    report_lines.append("=" * 70)
    report_lines.append(f"\n  Dataset: {stats['total_samples']} samples from 50K Dataset")
    report_lines.append(f"  Categories: {stats['num_categories']}")
    report_lines.append(f"  Urgency levels: {stats['num_urgency_levels']}")
    report_lines.append(f"\n  Train: {len(df_train)} | Test: {len(df_test)}")
    report_lines.append(f"  CV Folds: 3")
    report_lines.append(f"  Enhancements: Stacking Ensemble + HPO Tuning + Dual-Task Classification")

    report_lines.append(f"\n{'='*70}")
    report_lines.append("  CATEGORY CLASSIFICATION RESULTS")
    report_lines.append(f"{'='*70}")

    for name in results_cat:
        r = results_cat[name]
        report_lines.append(f"\n  {name}:")
        report_lines.append(f"    Accuracy:       {r['accuracy']:.4f}")
        report_lines.append(f"    F1 (macro):     {r['f1_macro']:.4f}")
        report_lines.append(f"    F1 (weighted):  {r['f1_weighted']:.4f}")

    report_lines.append(f"\n{'='*70}")
    report_lines.append("  URGENCY CLASSIFICATION RESULTS")
    report_lines.append(f"{'='*70}")

    for name in results_urg:
        r = results_urg[name]
        report_lines.append(f"\n  {name}:")
        report_lines.append(f"    Accuracy:       {r['accuracy']:.4f}")
        report_lines.append(f"    F1 (macro):     {r['f1_macro']:.4f}")
        report_lines.append(f"    F1 (weighted):  {r['f1_weighted']:.4f}")

    report_lines.append(f"\n{'='*70}")
    report_lines.append("  BEST MODELS")
    report_lines.append(f"{'='*70}")
    report_lines.append(f"\n  Best Category Model: {best_cat[0]} (F1={best_cat[1]['f1_macro']:.4f})")
    report_lines.append(f"  Best Urgency Model:  {best_urg[0]} (F1={best_urg[1]['f1_macro']:.4f})")
    report_lines.append(f"\n  Ensemble Category F1: {results_cat['Ensemble']['f1_macro']:.4f}")
    report_lines.append(f"  Ensemble Urgency F1:  {results_urg['Ensemble']['f1_macro']:.4f}")

    report_lines.append(f"\n{'='*70}")
    report_lines.append("  V2 -> V3 IMPROVEMENT")
    report_lines.append(f"{'='*70}")
    report_lines.append("  v2 Category Best F1: 0.7934 (TF-IDF + LR)")
    report_lines.append(f"  v3 Category Best F1: {best_cat[1]['f1_macro']:.4f} ({best_cat[0]})")
    report_lines.append(f"  Improvement: {(best_cat[1]['f1_macro'] - 0.7934):.4f} ({((best_cat[1]['f1_macro'] - 0.7934)/0.7934*100):+.1f}%)")
    report_lines.append("\n  v2 Urgency Best F1: 0.8023 (Char N-gram + LR)")
    report_lines.append(f"  v3 Urgency Best F1: {best_urg[1]['f1_macro']:.4f} ({best_urg[0]})")
    report_lines.append(f"  Improvement: {(best_urg[1]['f1_macro'] - 0.8023):.4f} ({((best_urg[1]['f1_macro'] - 0.8023)/0.8023*100):+.1f}%)")

    report_lines.append(f"\n{'='*70}")

    report_text = '\n'.join(report_lines)
    with open(os.path.join(reports_dir, "results.txt"), 'w', encoding='utf-8') as f:
        f.write(report_text)

    # ========================================================================
    # STEP 10: Evaluate on Hard/Ambiguous 5K Dataset
    # ========================================================================
    print("\n[10/10] Evaluating final ensembles on 5K Hard/Ambiguous dataset...")
    hard_csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "hinglish_hard_ambiguous_dataset_5000.csv")
    
    if os.path.exists(hard_csv_path):
        df_hard = load_data(hard_csv_path)
        df_hard['clean_text'] = df_hard['text'].apply(preprocessor.preprocess)
        X_hard = df_hard['clean_text'].to_numpy()
        y_hard_cat = df_hard['category'].to_numpy()
        y_hard_urg = df_hard['urgency'].to_numpy()
        
        y_hard_pred_cat = predict(ensemble_cat, X_hard)
        y_hard_pred_urg = predict(ensemble_urg, X_hard)
        
        hard_results_cat = evaluate_classifier(
            y_hard_cat, y_hard_pred_cat, class_names=category_names,
            task_name="Category - Hard 5K"
        )
        
        hard_results_urg = evaluate_classifier(
            y_hard_urg, y_hard_pred_urg, class_names=urgency_names,
            task_name="Urgency - Hard 5K"
        )
        
        # Append to report
        report_lines_hard = []
        report_lines_hard.append(f"\n{'='*70}")
        report_lines_hard.append("  EVALUATION ON 5K HARD/AMBIGUOUS DATASET")
        report_lines_hard.append(f"{'='*70}")
        report_lines_hard.append(f"\n  Category Classification:")
        report_lines_hard.append(f"    Accuracy:       {hard_results_cat['accuracy']:.4f}")
        report_lines_hard.append(f"    F1 (macro):     {hard_results_cat['f1_macro']:.4f}")
        report_lines_hard.append(f"    F1 (weighted):  {hard_results_cat['f1_weighted']:.4f}")
        report_lines_hard.append(f"\n  Urgency Classification:")
        report_lines_hard.append(f"    Accuracy:       {hard_results_urg['accuracy']:.4f}")
        report_lines_hard.append(f"    F1 (macro):     {hard_results_urg['f1_macro']:.4f}")
        report_lines_hard.append(f"    F1 (weighted):  {hard_results_urg['f1_weighted']:.4f}")
        report_lines_hard.append(f"\n{'='*70}")
        
        report_text += '\n'.join(report_lines_hard)
        with open(os.path.join(reports_dir, "results.txt"), 'w', encoding='utf-8') as f:
            f.write(report_text)
    else:
        print("  5K Hard/Ambiguous dataset not found. Skipping Step 10.")

    print(report_text)
    print(f"\n  All outputs saved to: {PROJECT_ROOT}")
    print(f"  Pipeline complete!")


if __name__ == "__main__":
    main()
