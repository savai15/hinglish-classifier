# Hinglish E-Commerce Complaint Classifier

## Overview
A text classification system that automatically routes Hinglish (Hindi-English code-mixed) customer complaints into support categories and flags their urgency level. Built for Indian e-commerce customer support teams to triage high-priority complaints faster.

## Problem
E-commerce customer support in India receives thousands of complaint messages daily written in Hinglish (e.g., "Order abhi tak nahi aaya, bahut urgent hai"). Standard NLP tools and pretrained embeddings perform poorly on this code-mixed, informally spelled input because words like "nahi", "nai", "bahot" are out-of-vocabulary for English-only models.

## Solution
This project builds a dual classification system that predicts:
- **Category**: Which type of complaint (Order Status, Delivery Issue, Wrong/Damaged Product, Returns & Refunds, Payment/Invoice, Account/Technical)
- **Urgency**: How urgent the complaint is (Low, Medium, High)

## Dataset
- **360 labeled Hinglish complaints** with real-world spelling variants
- **6 categories**: Account_Technical, Delivery_Issue, Order_Status, Payment_Invoice, Returns_Refunds, Wrong_Damaged_Product
- **3 urgency levels**: Low, Medium, High
- complaints exhibit natural code-mixing of Hindi and English in Roman script

## Approaches Compared

| Model | Category F1 | Urgency F1 |
|-------|-------------|------------|
| TF-IDF + Logistic Regression (Baseline) | 0.6607 | 0.6647 |
| Character N-gram + Logistic Regression | 0.6494 | **0.7230** |
| Combined Word + Char N-gram | **0.6916** | 0.6822 |
| TF-IDF + Linear SVM | 0.6856 | 0.6300 |

### Best Models
- **Category**: Combined Word + Char N-gram (F1 = 0.6916)
- **Urgency**: Character N-gram + LR (F1 = 0.7230)

### Key Finding
Character n-gram approach (mimicking FastText's subword method) outperforms word-level TF-IDF on Hinglish text because it naturally handles:
- Spelling variants (nahi/nai/nahee share character n-grams)
- Out-of-vocabulary words
- Code-mixed Hindi-English text

## Project Structure
```
project/
├── data/
│   ├── raw/                              # Original CSV dataset
│   └── processed/                        # Preprocessed data
├── models/                               # Trained models (.pkl files)
├── reports/                              # Visualizations and results
│   ├── category_cm_*.png                 # Confusion matrices
│   ├── category_model_comparison.png     # Model comparison chart
│   ├── category_per_class_f1.png         # Per-class F1 scores
│   ├── urgency_cm_*.png                  # Confusion matrices
│   ├── urgency_model_comparison.png      # Model comparison chart
│   └── results.txt                       # Full text report
├── src/
│   ├── __init__.py
│   ├── preprocessor.py                   # Hinglish text preprocessing
│   ├── data_loader.py                    # Data loading and splitting
│   ├── models.py                         # Model definitions
│   └── evaluation.py                     # Evaluation and visualization
├── main.py                               # End-to-end training pipeline
├── demo.py                               # Live demo with interactive input
└── requirements.txt                      # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords', quiet=True)"
```

## Usage

### Train All Models
```bash
cd project
python main.py
```

### Run Live Demo
```bash
python demo.py
```

### Use in Your Code
```python
from src.preprocessor import HinglishPreprocessor
from src.models import load_model

# Load
preprocessor = HinglishPreprocessor.load("models/preprocessor.pkl")
cat_model = load_model("models/category_combined_wordchar.pkl")
urg_model = load_model("models/urgency_char_n-gram__lr.pkl")

# Predict
complaint = "Mera order nahi aaya, refund do!"
cleaned = preprocessor.preprocess(complaint)
category = cat_model.predict([cleaned])[0]   # -> 'Order_Status'
urgency = urg_model.predict([cleaned])[0]    # -> 'High'
```

## Technical Details

### Preprocessing Pipeline
1. Lowercase conversion
2. Hinglish spelling normalization (nahi/nai/nahee -> nahin)
3. URL/email/phone/amount token replacement
4. Special character removal
5. Hindi + English stopword removal
6. Short word filtering

### Feature Extraction
- **Word-level TF-IDF**: Unigrams + bigrams, max 15K features
- **Character n-gram TF-IDF**: Character 2-5 grams, max 30K features (mimics FastText subword)
- **Combined**: Concatenated word + character features

### Classifiers
- Logistic Regression (with balanced class weights)
- Linear SVM (with calibration for probability estimates)

## Results Summary
- 360 labeled Hinglish complaints
- 4 model architectures compared
- Character n-gram approach shows advantage for code-mixed text
- Full pipeline trains in under 1 second
- All models, preprocessor, and visualizations saved to disk

## License
Educational project use only.
