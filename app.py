import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from extract_features import extract_lexical_features, FEATURE_NAMES

st.set_page_config(
    page_title="Phishing URL Detector & XAI Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Explainable & Generalizable Phishing URL Detector")
st.markdown("""
**Research System Prototype:** Real-time phishing URL detection leveraging **Reduced Lexical Features** 
combined with **Explainable AI (SHAP)**.
""")

# Load Models
MODEL_PATH = "models/phishing_detector.pkl"
EXPLAINER_PATH = "models/shap_explainer.pkl"

@st.cache_resource
def load_model_package():
    if not os.path.exists(MODEL_PATH):
        return None, None, FEATURE_NAMES
    artifact = joblib.load(MODEL_PATH)
    model = artifact['model']
    features = artifact.get('features', FEATURE_NAMES)
    explainer = joblib.load(EXPLAINER_PATH) if os.path.exists(EXPLAINER_PATH) else None
    return model, explainer, features

model, explainer, selected_features = load_model_package()

if model is None:
    st.warning("⚠️ Trained model not detected yet. Please run `python train_full_models.py` in your terminal to train and export the model artifact.")
    st.stop()

# Sidebar Preset URLs
st.sidebar.header("Test Quick Samples")
sample_choice = st.sidebar.selectbox(
    "Choose a preset test URL:",
    [
        "Custom Input",
        "https://www.google.com",
        "https://www.paypal.com/signin",
        "http://paypal-security-update-verify.com/login.php",
        "http://192.168.1.100/secure/bank/index.html",
        "http://verify-account-webscr.support-login-secure.info/signin",
        "https://en.wikipedia.org/wiki/Phishing"
    ]
)

default_val = "" if sample_choice == "Custom Input" else sample_choice
url_input = st.text_input("Enter URL to analyze:", value=default_val, placeholder="e.g. https://example-secure-login.com")

if st.button("🚀 Analyze URL", type="primary") and url_input.strip():
    with st.spinner("Analyzing lexical structure and computing SHAP values..."):
        # 1. Extract Features
        raw_feat = extract_lexical_features(url_input)
        df_feat = pd.DataFrame([raw_feat])[selected_features]

        # 2. Predict
        prob = model.predict_proba(df_feat)[0, 1] if hasattr(model, "predict_proba") else 0.5
        is_phish = prob >= 0.5
        confidence = prob if is_phish else (1.0 - prob)

        # 3. Display Main Verdict
        col1, col2, col3 = st.columns(3)
        with col1:
            if is_phish:
                st.error("### 🚨 VERDICT: PHISHING URL")
            else:
                st.success("### ✅ VERDICT: LEGITIMATE (SAFE)")
        with col2:
            st.metric("Model Confidence", f"{confidence * 100:.1f}%")
        with col3:
            st.metric("Phishing Probability", f"{prob * 100:.1f}%")

        # 4. Tabs for Details
        tab_shap, tab_features, tab_benchmark = st.tabs([
            "🔎 Explainable AI (SHAP Reasoning)", 
            "📋 Extracted Lexical Features", 
            "📊 Academic Benchmark Results"
        ])

        with tab_shap:
            st.subheader("Why did the AI make this decision?")
            if explainer is not None:
                try:
                    shap_vals = explainer.shap_values(df_feat)
                    if isinstance(shap_vals, list):
                        s_vals = shap_vals[1][0]
                    elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
                        s_vals = shap_vals[0, :, 1]
                    else:
                        s_vals = shap_vals[0]

                    shap_series = pd.Series(s_vals, index=selected_features).sort_values()
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    colors = ['#d9534f' if v > 0 else '#5cb85c' for v in shap_series.values]
                    shap_series.plot(kind='barh', ax=ax, color=colors)
                    ax.set_title("SHAP Feature Contributions (Red = Phishing Risk, Green = Safe Indicator)", fontsize=10)
                    ax.set_xlabel("SHAP Value (Impact on Prediction)")
                    ax.axvline(0, color='black', linestyle='--', alpha=0.7)
                    st.pyplot(fig)
                except Exception as e:
                    st.info(f"SHAP explanation calculation notice: {e}")
            else:
                st.info("SHAP explainer file not found. Run `train_full_models.py` to generate the explainer.")

        with tab_features:
            st.subheader(f"Extracted Reduced Features ({len(selected_features)} Features)")
            st.dataframe(df_feat.T.rename(columns={0: "Extracted Value"}))

        with tab_benchmark:
            st.subheader("Academic Benchmark Summary")
            if os.path.exists("data/model_benchmark_results.csv"):
                bench_df = pd.read_csv("data/model_benchmark_results.csv")
                st.dataframe(bench_df.set_index("Model"))
            else:
                st.info("Benchmark data table will appear after running `python train_full_models.py`.")
