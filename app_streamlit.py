"""
Streamlit Testing Dashboard for VR_05 Hinglish E-Commerce Complaint Classifier
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessor import HinglishPreprocessor

# Set page config
st.set_page_config(
    page_title="VR_05 Hinglish Complaint Classifier",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        border: 1px solid #FCA5A5;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        border: 1px solid #FCD34D;
    }
    .badge-low {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        border: 1px solid #6EE7B7;
    }
    .card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_vr05_models():
    """Cache models in memory for fast inference"""
    cat_model_path = os.path.join(PROJECT_ROOT, "models", "category_ensemble.pkl")
    urg_model_path = os.path.join(PROJECT_ROOT, "models", "urgency_ensemble.pkl")
    prep_path = os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl")

    if not os.path.exists(cat_model_path) or not os.path.exists(urg_model_path):
        st.error(f"⚠️ VR_05 Model files not found in `{os.path.join(PROJECT_ROOT, 'models')}`. Please ensure models are trained.")
        st.stop()

    cat_model = joblib.load(cat_model_path)
    urg_model = joblib.load(urg_model_path)
    preprocessor = HinglishPreprocessor()

    return cat_model, urg_model, preprocessor

# Load VR_05 Models
cat_model, urg_model, preprocessor = load_vr05_models()

# Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("VR_05 Architecture")
    st.markdown("**Production Specs:**")
    st.markdown("- **Classifier**: Stacking Ensemble (Linear SVM + LR)")
    st.markdown("- **Taxonomy**: 10 Categories | 3 Urgency Tiers")
    st.markdown("- **Features**: Dual Word + Subword Char TF-IDF")
    st.markdown("- **Hard Dataset F1**: `73.41%` Category | `85.0%` High Urgency Recall")
    st.markdown("---")
    st.caption("Engineered for CPU-only high-performance inference.")

# Header
st.markdown('<div class="main-header">🛍️ Hinglish E-Commerce Complaint Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time category categorization and urgency detection powered by VR_05 Stacking Ensemble</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["⚡ Single Complaint Testing", "📁 Batch CSV Testing"])

with tab1:
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("1. Enter Hinglish Complaint Text")
        
        # Sample Preset Buttons
        st.markdown("**Quick Preset Queries:**")
        preset_cols = st.columns(3)
        
        sample_query = ""
        if preset_cols[0].button("💳 Fraud/Legal Threat"):
            sample_query = "seller ne duplicate fake phone bhej diya, main consumer court jaunga police complaint karunga! refund mera immediately do!"
        if preset_cols[1].button("📦 Damaged Product"):
            sample_query = "box damaged tha aur andar screen tooti hui mili. refund window open karo please."
        if preset_cols[2].button("📱 App Bug"):
            sample_query = "checkout page load nahi ho raha, payment screen auto exit ho jaati hai bug fix karo app update mein."

        user_input = st.text_area(
            "Complaint Text (Hinglish / English / Code-Mixed):",
            value=sample_query if sample_query else "",
            height=140,
            placeholder="Type Hinglish complaint e.g., 'mera refund 10 din se pending hai jaldi karo else consumer court main report karunga'..."
        )

        analyze_btn = st.button("🚀 Analyze Complaint", type="primary", use_container_width=True)

    with col2:
        st.subheader("2. Real-Time Classification Results")
        
        if analyze_btn and user_input.strip():
            with st.spinner("Processing Hinglish text through VR_05 Stacking Meta-Learner..."):
                # Preprocess & Predict
                clean_text = preprocessor.preprocess(user_input)
                
                cat_pred = cat_model.predict([clean_text])[0]
                urg_pred = urg_model.predict([clean_text])[0]
                
                # Get probabilities
                cat_probs = cat_model.predict_proba([clean_text])[0]
                urg_probs = urg_model.predict_proba([clean_text])[0]
                
                cat_classes = cat_model.classes_
                urg_classes = urg_model.classes_

                cat_conf = np.max(cat_probs) * 100
                urg_conf = np.max(urg_probs) * 100

            # Result Cards
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # Category
            st.markdown(f"**Predicted Category:**")
            st.markdown(f"<h3 style='color:#0F172A; margin:0;'>📂 {cat_pred}</h3>", unsafe_allow_html=True)
            st.progress(int(cat_conf))
            st.caption(f"Category Confidence: **{cat_conf:.1f}%**")
            st.markdown("---")

            # Urgency
            st.markdown(f"**Assessed Urgency Level:**")
            if urg_pred == "High":
                st.markdown('<span class="badge-high">🚨 HIGH URGENCY</span>', unsafe_allow_html=True)
            elif urg_pred == "Medium":
                st.markdown('<span class="badge-medium">⚠️ MEDIUM URGENCY</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-low">🟢 LOW URGENCY</span>', unsafe_allow_html=True)
            
            st.caption(f"Urgency Confidence: **{urg_conf:.1f}%**")
            st.markdown('</div>', unsafe_allow_html=True)

            # Class Probabilities Breakdown
            st.markdown("**10-Category Probability Distribution:**")
            prob_df = pd.DataFrame({
                "Category": cat_classes,
                "Probability (%)": cat_probs * 100
            }).sort_values(by="Probability (%)", ascending=True)

            st.bar_chart(prob_df, x="Category", y="Probability (%)", color="#3B82F6", horizontal=True)

        elif analyze_btn:
            st.warning("⚠️ Please enter a Hinglish complaint text to analyze.")
        else:
            st.info("👈 Enter text or click a quick preset query on the left to test predictions.")

with tab2:
    st.subheader("📁 Batch CSV Classification")
    st.markdown("Upload a CSV file containing a column named `text` to run bulk classification.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        if "text" not in df_batch.columns:
            st.error("❌ CSV must contain a `text` column.")
        else:
            st.write(f"Loaded **{len(df_batch)}** rows.")
            if st.button("▶️ Run Batch Predictions"):
                with st.spinner("Classifying batch complaints..."):
                    df_batch['clean_text'] = df_batch['text'].astype(str).apply(preprocessor.preprocess)
                    df_batch['predicted_category'] = cat_model.predict(df_batch['clean_text'])
                    df_batch['predicted_urgency'] = urg_model.predict(df_batch['clean_text'])
                    
                    # Drop temp column
                    df_batch.drop(columns=['clean_text'], inplace=True)

                st.success("✅ Batch predictions complete!")
                st.dataframe(df_batch.head(20), use_container_width=True)

                # Download button
                csv_bytes = df_batch.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Classified CSV",
                    data=csv_bytes,
                    file_name="vr05_classified_complaints.csv",
                    mime="text/csv"
                )
