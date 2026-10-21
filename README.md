# 🔐 Phishing URL Detection Model

A machine learning-based web application that detects potentially malicious phishing URLs and classifies URLs as **Phishing** or **Legitimate**.

## 📌 Project Overview

Phishing attacks use deceptive URLs to steal sensitive information such as passwords, banking details, and personal data. This project uses machine learning and URL-based feature analysis to identify suspicious URLs and help users assess potential risks.

The project includes trained machine learning models, feature reduction, model evaluation, and SHAP-based explainability.

## ✨ Features

- 🔎 Predict whether a URL is phishing or legitimate.
- 🤖 Use a trained machine learning model for classification.
- 📊 Compare model performance using evaluation metrics.
- 🧠 Analyze feature importance and explain predictions using SHAP.
- 📉 Visualize ROC curves and feature reduction results.
- 🌐 Provide an application interface through Streamlit, if configured in `app.py`.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- SHAP
- Streamlit
- Matplotlib

## 📁 Project Structure

```text
Research/
├── app.py
├── predict_url.py
├── download_phiusiil.py
├── extract_features.py
├── reduce_features.py
├── train_baseline.py
├── train_full_models.py
├── requirements.txt
├── README.md
├── data/
│   ├── model_benchmark_results.csv
│   ├── reduced_features.json
│   └── sample_dataset.csv
├── models/
│   ├── phishing_detector.pkl
│   └── shap_explainer.pkl
├── feature_importance_ranking.png
├── feature_reduction_curve.png
├── results_metrics_comparison.png
├── results_roc_curves.png
├── shap_local_explanation.png
├── shap_summary.png
└── shap_summary_full.png
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/riyasachan94514/Phishing-URL-Detection.git
cd Phishing-URL-Detection
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

If `app.py` uses Streamlit, launch it with:

```bash
streamlit run app.py
```

## 🔍 Predict a URL

The project includes `predict_url.py` for URL prediction. Check the script for its expected input format and run instructions before using it.

## 📊 Model Evaluation

The project includes visualizations for:

- Model performance comparison
- ROC curves
- Feature importance ranking
- Feature reduction analysis
- SHAP explanations

See the PNG files in the repository for the corresponding results.

## 🧠 Explainable AI

SHAP (SHapley Additive exPlanations) helps interpret model predictions by showing how input features contribute to a prediction. This project includes SHAP summary and local explanation visualizations.

## ⚠️ Disclaimer

This project is intended for educational and research purposes. Predictions are not a guarantee that a URL is safe or malicious. Always use caution when opening unknown links.

## 👩‍💻 Author

**Riya Sachan**

GitHub: [@riyasachan94514](https://github.com/riyasachan94514)

---

⭐ If you find this project useful, consider giving the repository a star.