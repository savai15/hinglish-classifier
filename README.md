# Hinglish E-Commerce Complaint Classifier

## Overview
A text classification system that automatically routes Hinglish (Hindi-English code-mixed) customer complaints into support categories and flags their urgency level. Built for Indian e-commerce customer support teams to triage high-priority complaints faster.

## Problem
E-commerce customer support in India receives thousands of complaint messages daily written in Hinglish (e.g., "Order abhi tak nahi aaya, bahut urgent hai"). Standard NLP tools and pretrained embeddings perform poorly on this code-mixed, informally spelled input because words like "nahi", "nai", "bahot" are out-of-vocabulary for English-only models.

## Solution
This project builds a dual classification system that predicts:
- **Category**: Which type of complaint (Order Status, Delivery Issue, Wrong/Damaged Product, Returns & Refunds, Payment/Invoice, Account/Technical)
- **Urgency**: How urgent the complaint is (Low, Medium, High)

## Results

### Final Performance (v3 - Enhanced)

| Model | Category F1 | Urgency F1 |
|-------|-------------|------------|
| TF-IDF + Logistic Regression | 0.9265 | 0.9580 |
| Character N-gram + Logistic Regression | 0.9134 | 0.9582 |
| Combined Word + Char N-gram | 0.9065 | 0.9513 |
| TF-IDF + Linear SVM | **0.9332** | 0.9652 |
| **Ensemble (Top 3)** | 0.9263 | **0.9788** |

### Evolution Across Versions

| Version | Category F1 | Urgency F1 | Key Changes |
|---------|------------|------------|-------------|
| v1 (baseline) | 0.6916 | 0.7230 | 360 samples, no CV, no tuning |
| v2 (+CV, +tuning) | 0.7934 | 0.8023 | 5-fold CV, hyperparameter tuning, urgency cues |
| **v3 (+augmentation, +ensemble)** | **0.9332** | **0.9788** | 1002 samples, ensemble, confidence thresholding |

**Total improvement: Category +34.9% | Urgency +35.4%**

## Dataset
- **1002 labeled Hinglish complaints** (360 original + 642 synthetic via augmentation)
- **6 categories**: Account_Technical, Delivery_Issue, Order_Status, Payment_Invoice, Returns_Refunds, Wrong_Damaged_Product
- **3 urgency levels**: Low, Medium, High
- Complaints exhibit natural code-mixing of Hindi and English in Roman script

### Data Augmentation
- Generated 642 synthetic complaints using 180 urgency-specific templates
- Templates categorized by urgency level (HIGH/MEDIUM/LOW) with appropriate tone:
  - **HIGH**: Threats, escalation words, exclamation marks ("Consumer court jaunga!", "URGENT hai!")
  - **MEDIUM**: Firm complaints, issue descriptions ("Update do", "Please check karo")
  - **LOW**: Questions, general inquiries ("Usually kitne din leta h?", "Kaise karu?")
- Spelling variations applied to mimic real Hinglish writing patterns

## Enhancements (v1 -> v3)

### v2 Enhancements
1. **5-Fold Cross-Validation** - Reliable evaluation with mean +/- std reporting
2. **Enhanced Preprocessing** - Urgency cue detection (CAPS, threats, escalation, exclamation marks)
3. **Repeated Letter Normalization** - `urgenttt` -> `urgent`
4. **Hyperparameter Tuning** - RandomizedSearchCV with 25 iterations per model
5. **Error Analysis Module** - Confused pairs, per-class error rates, text length analysis

### v3 Enhancements
1. **Data Augmentation** - 360 -> 1002 samples with urgency-appropriate templates
2. **Ensemble Model** - Soft-voting of top 3 models per task
3. **Confidence Thresholding** - 40% threshold, flags low-confidence predictions for human review
4. **Updated Demo** - Shows confidence scores and [NEEDS REVIEW] flags

## Project Structure
```
project/
├── data/
│   ├── raw/
│   │   ├── hinglish_ecommerce_complaints_360_spelling_variants.csv   # Original 360 samples
│   │   └── hinglish_complaints_augmented.csv                         # Augmented 1002 samples
│   └── processed/                        # Preprocessed data
├── models/
│   ├── preprocessor.pkl                  # Hinglish text preprocessor
│   ├── category_ensemble.pkl             # Best category model (ensemble)
│   ├── category_tf-idf__svm.pkl          # Category TF-IDF + SVM
│   ├── category_tf-idf__lr.pkl           # Category TF-IDF + LR
│   ├── category_char_n-gram__lr.pkl      # Category Char N-gram + LR
│   ├── category_combined_wordchar.pkl    # Category Combined model
│   ├── urgency_ensemble.pkl              # Best urgency model (ensemble)
│   ├── urgency_tf-idf__svm.pkl           # Urgency TF-IDF + SVM
│   ├── urgency_tf-idf__lr.pkl            # Urgency TF-IDF + LR
│   ├── urgency_char_n-gram__lr.pkl       # Urgency Char N-gram + LR
│   └── urgency_combined_wordchar.pkl     # Urgency Combined model
├── reports/
│   ├── results.txt                       # Full text report
│   ├── category_cm_best.png              # Best category confusion matrix
│   ├── category_cm_norm_best.png         # Normalized confusion matrix
│   ├── category_model_comparison.png     # Model comparison chart
│   ├── category_per_class_f1.png         # Per-class F1 scores
│   ├── category_cv_comparison.png        # Cross-validation comparison
│   ├── category_error_rates.png          # Error rate comparison
│   ├── category_*_error_analysis.txt     # Detailed error analysis
│   ├── category_*_confused_pairs.png     # Confused class pairs
│   ├── category_*_per_class_accuracy.png # Per-class accuracy
│   ├── urgency_cm_best.png               # Best urgency confusion matrix
│   ├── urgency_cm_norm_best.png          # Normalized confusion matrix
│   ├── urgency_model_comparison.png      # Model comparison chart
│   ├── urgency_per_class_f1.png          # Per-class F1 scores
│   ├── urgency_cv_comparison.png         # Cross-validation comparison
│   ├── urgency_error_rates.png           # Error rate comparison
│   └── urgency_*_error_analysis.txt      # Detailed error analysis
├── src/
│   ├── __init__.py
│   ├── preprocessor.py                   # Hinglish text preprocessing (with urgency cues)
│   ├── data_loader.py                    # Data loading, splitting, CV splits
│   ├── models.py                         # Model definitions + ensemble + tuning
│   ├── evaluation.py                     # Evaluation, CV, visualizations
│   ├── error_analysis.py                 # Error analysis module
│   └── augment.py                        # Data augmentation generator
├── main.py                               # End-to-end training pipeline (v3)
├── demo.py                               # Live demo with confidence scores
├── app.py                                # Streamlit web app (beautiful UI)
└── requirements.txt                      # Python dependencies
```

## Installation

**No manual setup needed!** All scripts auto-install missing dependencies.

Just clone and run:
```bash
git clone https://github.com/savai15/test.git
cd test
streamlit run app.py
```

If you prefer manual installation:
```bash
pip install -r requirements.txt
```

## Usage

### Train All Models
```bash
cd project
python main.py
```

This will:
1. Load the augmented dataset (1002 samples)
2. Preprocess with urgency cue detection
3. Run 5-fold cross-validation on all models
4. Tune hyperparameters with RandomizedSearchCV
5. Build ensemble models
6. Evaluate on test set
7. Generate error analysis reports
8. Save all models and visualizations

### Run Streamlit Web App (Recommended)
```bash
cd project
streamlit run app.py
```
Opens a beautiful web interface at **http://localhost:8501** with:
- Dark gradient UI with glassmorphism design
- Real-time classification with confidence bars
- 8 clickable sample complaints
- Detailed probability tables
- Low-confidence warnings
- Sidebar with model info

### Run CLI Demo
```bash
python demo.py
```

Example output:
```
Complaint: Mera order abhi tak nahi aaya, bahut urgent hai!
-> Category: Order_Status (78%) | Urgency: High (96%)

Complaint: Profile name edit kar sakta hu??
-> Category: Account_Technical (79%) | Urgency: Low (93%)

Complaint: Consumer court me complaint karunga agar refund nahi mila!
-> Category: Returns_Refunds (60%) | Urgency: High (96%)
```

### Use in Your Code
```python
from src.preprocessor import HinglishPreprocessor
from src.models import load_model

# Load ensemble models
preprocessor = HinglishPreprocessor.load("models/preprocessor.pkl")
cat_model = load_model("models/category_ensemble.pkl")
urg_model = load_model("models/urgency_ensemble.pkl")

# Predict with confidence
import numpy as np

complaint = "Mera order nahi aaya, refund do!"
cleaned = preprocessor.preprocess(complaint)

category = cat_model.predict([cleaned])[0]
urgency = urg_model.predict([cleaned])[0]

# Get confidence scores
cat_proba = cat_model.predict_proba([cleaned])[0]
urg_proba = urg_model.predict_proba([cleaned])[0]
cat_confidence = np.max(cat_proba)
urg_confidence = np.max(urg_proba)

# Flag low confidence predictions
THRESHOLD = 0.4
if cat_confidence < THRESHOLD:
    print("Category prediction needs human review")
if urg_confidence < THRESHOLD:
    print("Urgency prediction needs human review")
```

### Generate Augmented Data
```bash
python src/augment.py
```

## Technical Details

### Preprocessing Pipeline (Enhanced v2)
1. Lowercase conversion
2. Repeated letter normalization (`urgenttt` -> `urgent`)
3. Hinglish spelling normalization (`nahi/nai/nahee` -> `nahin`)
4. **Urgency cue detection** (before cleaning):
   - ALL CAPS words -> `URGENTCAPS` token
   - Exclamation marks -> `EXCLAMATION` / `EXCLAMATIONHIGH` tokens
   - Threat words ("consumer court", "legal") -> `THREAT` token
   - Escalation words ("manager", "senior") -> `ESCALATION` token
   - High-value amounts (>=5000) -> `HIGHAMOUNT` token
   - Urgency keywords ("urgent", "jaldi") -> `URGENTKEYWORD` token
   - Time pressure words -> `TIMEPRESSURE` token
5. URL/email/phone/amount token replacement
6. Special character removal
7. Hindi + English stopword removal
8. Short word filtering
9. Urgency tokens appended to text

### Feature Extraction
- **Word-level TF-IDF**: Unigrams + bigrams, tuned max features
- **Character n-gram TF-IDF**: Character 2-6 grams, tuned max features (mimics FastText subword)
- **Combined**: Concatenated word + character features

### Classifiers
- Logistic Regression (with balanced class weights, tuned C)
- Linear SVM (with calibration for probability estimates, tuned C)
- **Ensemble**: Soft-voting of top 3 models per task

### Hyperparameter Tuning
- RandomizedSearchCV with 25 iterations, 3-fold CV
- Tuned parameters: `max_features`, `ngram_range`, `min_df`, `max_df`, `C`

### Cross-Validation
- 5-fold stratified cross-validation
- Reports mean F1 with standard deviation

### Confidence Thresholding
- 40% confidence threshold
- Predictions below threshold flagged as `[NEEDS REVIEW]`

## Error Analysis

### Category Classification
- **Account_Technical** and **Wrong_Damaged_Product**: 0% error rate (perfect)
- **Order_Status**: 16% error rate (most confused with Account_Technical)
- Most common confused pairs: Order_Status <-> Account_Technical

### Urgency Classification
- **High urgency**: 0% error rate (perfect)
- **Low urgency**: 3.8% error rate
- **Medium urgency**: 2.4% error rate
- Most common confused pairs: Low <-> Medium

## License
Educational project use only.
