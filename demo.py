"""
Hinglish E-Commerce Complaint Classifier - Live Demo (Enhanced v2)
=================================================================
Run this script to test the trained models with custom Hinglish complaints.
Now with confidence scores and urgency cue detection.
"""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessor import HinglishPreprocessor
from src.models import load_model


def get_confidence(pipeline, text):
    """Get prediction confidence (probability) if available."""
    proba = pipeline.predict_proba([text])[0]
    return float(np.max(proba))


def main():
    print("=" * 70)
    print("  HINGLISH COMPLAINT CLASSIFIER - LIVE DEMO (Enhanced v2)")
    print("=" * 70)

    # Load models
    print("\n  Loading models...")
    preprocessor = HinglishPreprocessor.load(os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl"))
    cat_model = load_model(os.path.join(PROJECT_ROOT, "models", "category_combined_wordchar.pkl"))
    urg_model = load_model(os.path.join(PROJECT_ROOT, "models", "urgency_char_ngram__lr.pkl"))
    print("  Models loaded.\n")

    # Sample Hinglish complaints for demo
    demo_complaints = [
        "Mera order abhi tak nahi aaya, bahut urgent hai!",
        "Refund kab milega? 10 din ho gaye, paisa wapas karo!",
        "Wrong product bheja hai, exchange karo jaldi se",
        "Payment fail ho gaya but paisa kat gaya, ab kya karu?",
        "App crash ho raha hai, login nahi ho raha",
        "Delivery boy ne parcel fek ke diya, bahut damage hai",
        "Mera paisa double charge hua hai, refund do!",
        "Order status update nahi ho raha, kya problem hai?",
        "Invoice me GST number galat hai, correct karo",
        "Package missing hai, track karo jaldi!",
        "Consumer court me complaint karunga agar refund nahi mila!",
        "Manager se baat karao, tumse nahi ho raha",
    ]

    print("-" * 70)
    print("  DEMO: Classifying sample Hinglish complaints")
    print("-" * 70)

    for i, complaint in enumerate(demo_complaints, 1):
        cleaned = preprocessor.preprocess(complaint)
        category = cat_model.predict([cleaned])[0]
        urgency = urg_model.predict([cleaned])[0]

        cat_conf = get_confidence(cat_model, cleaned)
        urg_conf = get_confidence(urg_model, cleaned)

        print(f"\n  [{i}] Complaint: {complaint}")
        print(f"      Cleaned:  {cleaned[:60]}...")
        print(f"      Category: {category} (confidence: {cat_conf:.2f})")
        print(f"      Urgency:  {urgency} (confidence: {urg_conf:.2f})")

    print("\n" + "-" * 70)
    print("  INTERACTIVE MODE")
    print("-" * 70)
    print("  Type a Hinglish complaint below (or 'quit' to exit):\n")

    while True:
        try:
            user_input = input("  > ").strip()
            if user_input.lower() in ['quit', 'exit', 'q', '']:
                print("  Goodbye!")
                break

            cleaned = preprocessor.preprocess(user_input)
            category = cat_model.predict([cleaned])[0]
            urgency = urg_model.predict([cleaned])[0]

            cat_conf = get_confidence(cat_model, cleaned)
            urg_conf = get_confidence(urg_model, cleaned)

            print(f"    Cleaned:  {cleaned[:60]}...")
            print(f"    Category: {category} (confidence: {cat_conf:.2f})")
            print(f"    Urgency:  {urgency} (confidence: {urg_conf:.2f})\n")

        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye!")
            break


if __name__ == "__main__":
    main()
