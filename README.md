# Hinglish E-Commerce Complaint Classifier (v4.01 - VR_05)

## Overview
An advanced, dual-task text classification system designed to automatically categorize and assess the urgency of Hinglish (Hindi-English code-mixed) customer complaints in Indian e-commerce customer support workflows.

---

## Model Architecture (VR_05 Specification)

### 1. Two-Stage Stacking Meta-Learner Ensemble
- **Base Estimators**:
  - `TF-IDF + Logistic Regression`
  - `Character N-gram + Logistic Regression`
  - `Combined Word + Character N-Gram`
  - `TF-IDF + Linear SVM`
- **Meta-Learner**: `Logistic Regression` trained on out-of-fold probability predictions.

### 2. Anti-Shortcut Formatting Normalization
- Exclamation marks (`!`, `!!`), question marks (`?`), and ALL-CAPS words (`URGENT`, `PLEASE`, `ATTENTION`, `FYI`) are **randomly distributed across all 3 urgency levels** to prevent the classifier from relying on formatting shortcuts and force it to evaluate true semantic intent.

### 3. Leakage-Free Group Validation
- Splits data using `GroupKFold` and `GroupShuffleSplit` on synthetic seed groups, ensuring similar/duplicate template variations never leak into held-out test splits.

---

## Taxonomy & Dataset

- **Dataset**: `data/raw/hinglish_dataset_50000_v2.csv` (50,000 balanced samples)
- **10 Complaint Categories**:
  1. `App_Bug`
  2. `Billing_Invoice`
  3. `Customer_Service`
  4. `Damaged_Product`
  5. `Late_Delivery`
  6. `Order_Not_Delivered`
  7. `Payment_Issue`
  8. `Refund_Return`
  9. `Seller_Fraud`
  10. `Wrong_Product`
- **3 Urgency Tiers**: `High`, `Medium`, `Low`

---

## Performance Benchmarks

### Held-Out Test Set (7,412 samples)
| Task | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| **Category Classification** | 1.0000 | **1.0000** | 1.0000 |
| **Urgency Classification** | 1.0000 | **1.0000** | 1.0000 |

### Ground-Truth Evaluation on 5,000 Hard/Ambiguous Real Complaints
| Task / Metric | Score | Key Highlight |
|---|---|---|
| **Category Weighted F1** | **73.41%** | Top classes: Refund_Return (0.91 F1), Customer_Service (0.90 F1) |
| **Urgency High-Urgency Recall** | **85.00%** | Catches 85% of critical financial loss / legal threat complaints |
| **Urgency Weighted F1** | **68.14%** | Medium Urgency Precision: 96% |

---

## Project Structure
```
project/
├── data/
│   ├── raw/
│   │   ├── hinglish_dataset_50000_v2.csv                       # 50K v2 dataset
│   │   └── hinglish_hard_ambiguous_dataset_5000_relabeled.csv  # Relabeled 5K hard dataset
├── models/
│   ├── category_ensemble.pkl                                   # Stacking ensemble for category
│   ├── urgency_ensemble.pkl                                    # Stacking ensemble for urgency
│   └── preprocessor.pkl                                        # Hinglish preprocessor
├── src/
│   ├── augment.py                                              # 50K Dataset Generator (v2)
│   ├── data_loader.py                                          # Group-aware data loader
│   ├── preprocessor.py                                         # Text normalization
│   ├── models.py                                               # Stacking Ensemble implementation
│   └── evaluation.py                                           # Evaluation suite
├── main.py                                                     # Pipeline execution script
└── relabel_5k_hard_dataset.py                                  # Semantic urgency relabeler
```
