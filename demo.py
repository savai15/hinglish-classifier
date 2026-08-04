"""
Hinglish E-Commerce Complaint Classifier - Live Demo (Enhanced v3)
=================================================================
Run this script to test the trained models with custom Hinglish complaints.
Now with ensemble models, confidence scores, and thresholding.
"""
import subprocess
import sys

# Auto-install missing dependencies
REQUIRED_PACKAGES = {
    'numpy': 'numpy>=1.24.0',
    'pandas': 'pandas>=2.0.0',
    'sklearn': 'scikit-learn>=1.3.0',
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
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessor import HinglishPreprocessor
from src.models import load_model

CONFIDENCE_THRESHOLD = 0.4


def get_prediction_with_confidence(model, text):
    """Get prediction with confidence score and flag if low confidence."""
    proba = model.predict_proba([text])[0]
    pred_idx = np.argmax(proba)
    confidence = float(proba[pred_idx])
    prediction = model.classes_[pred_idx]
    is_low_confidence = confidence < CONFIDENCE_THRESHOLD

    return prediction, confidence, is_low_confidence


def main():
    print("=" * 70)
    print("  HINGLISH COMPLAINT CLASSIFIER - LIVE DEMO (Enhanced v3)")
    print("  Ensemble Models + Confidence Scores + Thresholding")
    print("=" * 70)

    # Load models
    print("\n  Loading models...")
    preprocessor = HinglishPreprocessor.load(os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl"))

    # Try ensemble first, fallback to best single model
    try:
        cat_model = load_model(os.path.join(PROJECT_ROOT, "models", "category_ensemble.pkl"))
        print("    Loaded: Category Ensemble")
    except FileNotFoundError:
        cat_model = load_model(os.path.join(PROJECT_ROOT, "models", "category_tf-idf__lr.pkl"))
        print("    Loaded: Category TF-IDF + LR (ensemble not found)")

    try:
        urg_model = load_model(os.path.join(PROJECT_ROOT, "models", "urgency_ensemble.pkl"))
        print("    Loaded: Urgency Ensemble")
    except FileNotFoundError:
        urg_model = load_model(os.path.join(PROJECT_ROOT, "models", "urgency_char_ngram__lr.pkl"))
        print("    Loaded: Urgency Char N-gram + LR (ensemble not found)")

    print("  Models loaded.\n")

    # Sample complaints for demo
    demo_complaints = [
        ("Mera order abhi tak nahi aaya, bahut urgent hai!", "Expected: Order_Status + High"),
        ("Refund kab milega? 10 din ho gaye, paisa wapas karo!", "Expected: Returns_Refunds + High"),
        ("Wrong product bheja hai, exchange karo jaldi se", "Expected: Wrong_Damaged_Product + High"),
        ("Payment fail ho gaya but paisa kat gaya, ab kya karu?", "Expected: Payment_Invoice + Medium"),
        ("App crash ho raha hai, login nahi ho raha", "Expected: Account_Technical + Medium"),
        ("Delivery boy ne parcel fek ke diya, bahut damage hai", "Expected: Delivery_Issue + High"),
        ("Mera paisa double charge hua hai, refund do!", "Expected: Payment_Invoice + High"),
        ("Order status update nahi ho raha, kya problem hai?", "Expected: Order_Status + Medium"),
        ("Invoice me GST number galat hai, correct karo", "Expected: Payment_Invoice + Low"),
        ("Package missing hai, track karo jaldi!", "Expected: Delivery_Issue + High"),
        ("Consumer court me complaint karunga agar refund nahi mila!", "Expected: Returns_Refunds + High"),
        ("Manager se baat karao, tumse nahi ho raha", "Expected: Order_Status + High"),
        ("Refund usually kitne working days leta h?", "Expected: Returns_Refunds + Low"),
        ("Profile name edit kar sakta hu??", "Expected: Account_Technical + Low"),
    ]

    print("-" * 70)
    print("  DEMO: Classifying sample Hinglish complaints")
    print("-" * 70)

    for i, (complaint, expected) in enumerate(demo_complaints, 1):
        cleaned = preprocessor.preprocess(complaint)
        category, cat_conf, cat_low = get_prediction_with_confidence(cat_model, cleaned)
        urgency, urg_conf, urg_low = get_prediction_with_confidence(urg_model, cleaned)

        cat_flag = " [LOW]" if cat_low else ""
        urg_flag = " [LOW]" if urg_low else ""

        print(f"\n  [{i}] {complaint}")
        print(f"      Expected:  {expected}")
        print(f"      Category:  {category} ({cat_conf:.0%}){cat_flag}")
        print(f"      Urgency:   {urgency} ({urg_conf:.0%}){urg_flag}")

    print("\n" + "-" * 70)
    print("  INTERACTIVE MODE")
    print("-" * 70)
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
    print(f"  [LOW] = Below threshold, needs human review")
    print("  Type a Hinglish complaint below (or 'quit' to exit):\n")

    while True:
        try:
            user_input = input("  > ").strip()
            if user_input.lower() in ['quit', 'exit', 'q', '']:
                print("  Goodbye!")
                break

            cleaned = preprocessor.preprocess(user_input)
            category, cat_conf, cat_low = get_prediction_with_confidence(cat_model, cleaned)
            urgency, urg_conf, urg_low = get_prediction_with_confidence(urg_model, cleaned)

            cat_flag = " [NEEDS REVIEW]" if cat_low else ""
            urg_flag = " [NEEDS REVIEW]" if urg_low else ""

            print(f"    Category:  {category} ({cat_conf:.0%}){cat_flag}")
            print(f"    Urgency:   {urgency} ({urg_conf:.0%}){urg_flag}\n")

        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye!")
            break


if __name__ == "__main__":
    main()
