# Phishing URL Detection: Full Research Pipeline

This repository implements the complete system for the paper:  
**"Explainable and Generalizable Phishing URL Detection Using Reduced URL Features"**.

---

## 📁 Repository Structure

```
├── data/
│   ├── sample_dataset.csv             # Fallback simulated dataset
│   ├── phiusiil_dataset.csv           # Cleaned real PhiUSIIL dataset (235k rows)
│   ├── reduced_features.json          # Selected reduced feature config
│   └── model_benchmark_results.csv    # Final comparative benchmark table
├── models/
│   ├── phishing_detector.pkl          # Exported trained production model
│   └── shap_explainer.pkl             # Exported SHAP TreeExplainer
├── extract_features.py                # Fast lexical feature extractor (12 features)
├── download_phiusiil.py               # Official UCI PhiUSIIL dataset loader
├── reduce_features.py                 # Feature selection & ablation study (MI + RF)
├── train_full_models.py               # Multi-model benchmark (RF, XGBoost, DL + SHAP)
├── predict_url.py                     # Single URL prediction & CLI XAI report
├── app.py                             # Interactive Streamlit Web Dashboard
├── requirements.txt                   # Project dependencies
└── README.md                          # Documentation
```

---

## 🚀 Step-by-Step Execution Guide

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: (Optional) Fetch Official UCI PhiUSIIL Dataset
To download the official UCI benchmark (235,795 URLs):
```powershell
python download_phiusiil.py
```
*(If you skip this step, all scripts automatically fallback to the built-in sample dataset for immediate testing).*

### Step 3: Run Feature Reduction & Selection
Calculates Mutual Information (MI) and Random Forest feature importance, performs an ablation study, and saves the top reduced features:
```powershell
python reduce_features.py
```
*Outputs: `feature_importance_ranking.png` and `feature_reduction_curve.png`.*

### Step 4: Train Comparative Models (ML vs. DL) & Compute SHAP
Trains Random Forest, XGBoost, and Deep Neural Net (MLP), calculates accuracy, F1, and latency, and generates publication plots:
```powershell
python train_full_models.py
```
*Outputs: `results_roc_curves.png`, `results_metrics_comparison.png`, and `shap_summary_full.png`.*

### Step 5: Test Any URL from the Terminal
Test any link in milliseconds with human-readable explanations:
```powershell
python predict_url.py "http://paypal-security-update-verify.com/login.php"
```

### Step 6: Launch the Interactive Web Dashboard
Run the Streamlit web dashboard in your browser:
```powershell
streamlit run app.py
```
