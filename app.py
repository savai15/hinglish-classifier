"""
Hinglish E-Commerce Complaint Classifier - Streamlit Web App
=============================================================
Interactive web interface for classifying Hinglish complaints.
Run with: streamlit run app.py
"""
import os
import sys
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessor import HinglishPreprocessor
from src.models import load_model

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Hinglish Complaint Classifier",
    page_icon=" Indian Flag",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 3rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
    }

    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        border-color: rgba(102, 126, 234, 0.5);
    }

    .metric-card h3 {
        color: rgba(255,255,255,0.7);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }

    .metric-card .value {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
    }

    .metric-card .label {
        color: rgba(255,255,255,0.5);
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }

    /* Prediction cards */
    .prediction-card {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
    }

    .prediction-result {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }

    .prediction-result.high {
        background: linear-gradient(135deg, rgba(255, 71, 87, 0.2) 0%, rgba(255, 107, 107, 0.1) 100%);
        border-left: 4px solid #ff4757;
    }

    .prediction-result.medium {
        background: linear-gradient(135deg, rgba(255, 165, 2, 0.2) 0%, rgba(255, 183, 77, 0.1) 100%);
        border-left: 4px solid #ffa502;
    }

    .prediction-result.low {
        background: linear-gradient(135deg, rgba(46, 213, 115, 0.2) 0%, rgba(46, 213, 115, 0.1) 100%);
        border-left: 4px solid #2ed573;
    }

    .prediction-result.category {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-left: 4px solid #667eea;
    }

    .result-label {
        color: rgba(255,255,255,0.6);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-value {
        color: white;
        font-size: 1.5rem;
        font-weight: 600;
    }

    .result-confidence {
        color: rgba(255,255,255,0.5);
        font-size: 0.9rem;
    }

    /* Sample complaint cards */
    .sample-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .sample-card:hover {
        background: rgba(102, 126, 234, 0.15);
        border-color: rgba(102, 126, 234, 0.4);
        transform: translateX(5px);
    }

    .sample-card .text {
        color: rgba(255,255,255,0.9);
        font-size: 0.95rem;
    }

    .sample-card .meta {
        color: rgba(255,255,255,0.4);
        font-size: 0.8rem;
        margin-top: 0.3rem;
    }

    /* Confidence bar */
    .confidence-bar {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        height: 8px;
        margin-top: 0.5rem;
        overflow: hidden;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }

    .confidence-fill.high { background: linear-gradient(90deg, #2ed573, #7bed9f); }
    .confidence-fill.medium { background: linear-gradient(90deg, #ffa502, #ffbe76); }
    .confidence-fill.low { background: linear-gradient(90deg, #ff4757, #ff6b81); }

    /* Urgency badges */
    .urgency-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .urgency-badge.high {
        background: linear-gradient(135deg, #ff4757, #ff6b81);
        color: white;
    }

    .urgency-badge.medium {
        background: linear-gradient(135deg, #ffa502, #ffbe76);
        color: white;
    }

    .urgency-badge.low {
        background: linear-gradient(135deg, #2ed573, #7bed9f);
        color: white;
    }

    /* Category badges */
    .category-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }

    /* Info cards */
    .info-card {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
    }

    .info-card h4 {
        color: #667eea;
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
    }

    .info-card p {
        color: rgba(255,255,255,0.7);
        margin: 0;
        font-size: 0.9rem;
    }

    /* Low confidence warning */
    .low-confidence-warning {
        background: linear-gradient(135deg, rgba(255, 71, 87, 0.15) 0%, rgba(255, 107, 107, 0.08) 100%);
        border: 1px solid rgba(255, 71, 87, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }

    .low-confidence-warning .icon {
        font-size: 1.5rem;
    }

    .low-confidence-warning .text {
        color: rgba(255,255,255,0.9);
        font-size: 0.95rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: white;
    }

    /* Text input styling */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        color: white !important;
        font-size: 1.1rem !important;
    }

    .stTextArea textarea:focus {
        border-color: rgba(102, 126, 234, 0.5) !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
        color: white !important;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin: 2rem 0;
    }

    /* Remove default streamlit padding */
    .block-container {
        padding-top: 2rem !important;
    }

    /* Table styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# LOAD MODELS (cached)
# ============================================================================
@st.cache_resource
def load_models():
    """Load all models (cached for performance)."""
    preprocessor = HinglishPreprocessor.load(os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl"))

    try:
        cat_model = load_model(os.path.join(PROJECT_ROOT, "models", "category_ensemble.pkl"))
        cat_model_name = "Ensemble (TF-IDF + SVM + LR)"
    except FileNotFoundError:
        cat_model = load_model(os.path.join(PROJECT_ROOT, "models", "category_tf-idf__svm.pkl"))
        cat_model_name = "TF-IDF + SVM"

    try:
        urg_model = load_model(os.path.join(PROJECT_ROOT, "models", "urgency_ensemble.pkl"))
        urg_model_name = "Ensemble (TF-IDF + SVM + LR)"
    except FileNotFoundError:
        urg_model = load_model(os.path.join(PROJECT_ROOT, "models", "urgency_char_n-gram__lr.pkl"))
        urg_model_name = "Char N-gram + LR"

    return preprocessor, cat_model, cat_model_name, urg_model, urg_model_name


def get_prediction(model, text):
    """Get prediction with confidence."""
    proba = model.predict_proba([text])[0]
    pred_idx = np.argmax(proba)
    prediction = model.classes_[pred_idx]
    confidence = float(proba[pred_idx])

    # Get all class probabilities
    all_probs = {cls: float(prob) for cls, prob in zip(model.classes_, proba)}

    return prediction, confidence, all_probs


# ============================================================================
# SAMPLE COMPLAINTS
# ============================================================================
SAMPLE_COMPLAINTS = [
    {"text": "Mera order abhi tak nahi aaya, bahut urgent hai!", "category": "Order_Status", "urgency": "High"},
    {"text": "Refund kab milega? 10 din ho gaye, paisa wapas karo!", "category": "Returns_Refunds", "urgency": "High"},
    {"text": "Wrong product bheja hai, exchange karo jaldi se", "category": "Wrong_Damaged_Product", "urgency": "High"},
    {"text": "Payment fail ho gaya but paisa kat gaya, ab kya karu?", "category": "Payment_Invoice", "urgency": "Medium"},
    {"text": "App crash ho raha hai, login nahi ho raha", "category": "Account_Technical", "urgency": "Medium"},
    {"text": "Consumer court me complaint karunga agar refund nahi mila!", "category": "Returns_Refunds", "urgency": "High"},
    {"text": "Profile name edit kar sakta hu??", "category": "Account_Technical", "urgency": "Low"},
    {"text": "Refund usually kitne working days leta h?", "category": "Returns_Refunds", "urgency": "Low"},
    {"text": "Delivery boy ne package fek ke diya, bahut damage hai", "category": "Delivery_Issue", "urgency": "High"},
    {"text": "Order status update nahi ho raha, kya problem hai?", "category": "Order_Status", "urgency": "Medium"},
    {"text": "Invoice me GST number galat hai, correct karo", "category": "Payment_Invoice", "urgency": "Low"},
    {"text": "Package missing hai, track karo jaldi!", "category": "Delivery_Issue", "urgency": "High"},
    {"text": "Manager se baat karao, tumse nahi ho raha", "category": "Order_Status", "urgency": "High"},
    {"text": "Mera paisa double charge hua hai, refund do!", "category": "Payment_Invoice", "urgency": "High"},
    {"text": "Delivery instructions follow nahi kiye, gate pe chhod ke chala gaya", "category": "Delivery_Issue", "urgency": "Medium"},
]


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Load models
    preprocessor, cat_model, cat_model_name, urg_model, urg_model_name = load_models()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Hinglish Complaint Classifier</h1>
        <p>AI-powered classification of Hinglish e-commerce complaints into categories and urgency levels</p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Dataset</h3>
            <div class="value">1,002</div>
            <div class="label">Labeled Complaints</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Category F1</h3>
            <div class="value">93.3%</div>
            <div class="label">TF-IDF + SVM</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Urgency F1</h3>
            <div class="value">97.9%</div>
            <div class="label">Ensemble Model</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Categories</h3>
            <div class="value">6 + 3</div>
            <div class="label">Types + Urgency Levels</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Main layout
    col_input, col_results = st.columns([1, 1])

    with col_input:
        st.markdown("### Enter Complaint")

        # Text area
        complaint_text = st.text_area(
            "Type a Hinglish complaint:",
            placeholder="e.g., Mera order abhi tak nahi aaya, bahut urgent hai!",
            height=120,
            label_visibility="collapsed",
        )

        # Classify button
        classify_clicked = st.button("Classify Complaint", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Sample complaints
        st.markdown("### Try Sample Complaints")
        st.markdown('<p style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">Click a sample to classify it:</p>', unsafe_allow_html=True)

        for i, sample in enumerate(SAMPLE_COMPLAINTS[:8]):
            urgency_class = sample["urgency"].lower()
            if st.button(
                f"{sample['text'][:55]}{'...' if len(sample['text']) > 55 else ''}",
                key=f"sample_{i}",
                use_container_width=True,
            ):
                complaint_text = sample["text"]
                classify_clicked = True

    with col_results:
        st.markdown("### Classification Results")

        if classify_clicked and complaint_text.strip():
            # Preprocess
            cleaned = preprocessor.preprocess(complaint_text)

            # Get predictions
            category, cat_conf, cat_all_probs = get_prediction(cat_model, cleaned)
            urgency, urg_conf, urg_all_probs = get_prediction(urg_model, cleaned)

            # Check confidence
            cat_low = cat_conf < 0.4
            urg_low = urg_conf < 0.4

            # Low confidence warning
            if cat_low or urg_low:
                warning_parts = []
                if cat_low:
                    warning_parts.append("category")
                if urg_low:
                    warning_parts.append("urgency")
                warning_text = " and ".join(warning_parts)

                st.markdown(f"""
                <div class="low-confidence-warning">
                    <div class="icon">&#9888;</div>
                    <div class="text">Low confidence on {warning_text} prediction(s). Consider human review.</div>
                </div>
                """, unsafe_allow_html=True)

            # Category result
            cat_urgency_class = "high" if cat_conf >= 0.7 else "medium" if cat_conf >= 0.5 else "low"
            st.markdown(f"""
            <div class="prediction-result category">
                <div>
                    <div class="result-label">Category</div>
                    <div class="result-value">{category.replace('_', ' ')}</div>
                    <div class="result-confidence">Confidence: {cat_conf:.1%}</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill {cat_urgency_class}" style="width: {cat_conf*100}%"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Urgency result
            urg_class = urgency.lower()
            st.markdown(f"""
            <div class="prediction-result {urg_class}">
                <div>
                    <div class="result-label">Urgency Level</div>
                    <div class="result-value">{urgency}</div>
                    <div class="result-confidence">Confidence: {urg_conf:.1%}</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill {urg_class}" style="width: {urg_conf*100}%"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Detailed probabilities
            st.markdown("<br>", unsafe_allow_html=True)

            with st.expander("View Detailed Probabilities", expanded=False):
                prob_col1, prob_col2 = st.columns(2)

                with prob_col1:
                    st.markdown("**Category Probabilities**")
                    cat_probs_df = pd.DataFrame([
                        {"Category": k.replace('_', ' '), "Probability": v}
                        for k, v in sorted(cat_all_probs.items(), key=lambda x: -x[1])
                    ])
                    st.dataframe(cat_probs_df, use_container_width=True, hide_index=True)

                with prob_col2:
                    st.markdown("**Urgency Probabilities**")
                    urg_probs_df = pd.DataFrame([
                        {"Urgency": k, "Probability": v}
                        for k, v in sorted(urg_all_probs.items(), key=lambda x: -x[1])
                    ])
                    st.dataframe(urg_probs_df, use_container_width=True, hide_index=True)

            # Preprocessed text
            with st.expander("View Preprocessed Text", expanded=False):
                st.code(cleaned, language=None)

        elif classify_clicked and not complaint_text.strip():
            st.warning("Please enter a complaint to classify.")

        else:
            st.markdown("""
            <div class="info-card">
                <h4>How to use</h4>
                <p>Enter a Hinglish complaint on the left and click "Classify Complaint" to see the predicted category and urgency level.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-card">
                <h4>Supported Categories</h4>
                <p>
                <span class="category-badge">Order Status</span>
                <span class="category-badge">Delivery Issue</span>
                <span class="category-badge">Wrong/Damaged Product</span><br><br>
                <span class="category-badge">Returns/Refunds</span>
                <span class="category-badge">Payment/Invoice</span>
                <span class="category-badge">Account/Technical</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-card">
                <h4>Urgency Levels</h4>
                <p>
                <span class="urgency-badge high">High</span> - Threats, escalation, immediate action needed<br>
                <span class="urgency-badge medium">Medium</span> - Firm complaints, issue descriptions<br>
                <span class="urgency-badge low">Low</span> - Questions, general inquiries
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## Model Information")

        st.markdown(f"""
        <div class="info-card">
            <h4>Category Model</h4>
            <p>{cat_model_name}<br>Test F1: 0.9332</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-card">
            <h4>Urgency Model</h4>
            <p>{urg_model_name}<br>Test F1: 0.9788</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("## Features")
        st.markdown("""
        - 5-fold cross-validation
        - Hyperparameter tuning
        - Ensemble of top 3 models
        - Urgency cue detection
        - Confidence thresholding
        - Spelling normalization
        """)

        st.markdown("---")

        st.markdown("## Tech Stack")
        st.markdown("""
        - **Python** - Core language
        - **Scikit-learn** - ML models
        - **TF-IDF** - Feature extraction
        - **Streamlit** - Web interface
        """)

        st.markdown("---")

        st.markdown("## Links")
        st.markdown("[GitHub Repository](https://github.com/savai15/test)")


if __name__ == "__main__":
    main()
