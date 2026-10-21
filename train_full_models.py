import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report
)
import shap

from extract_features import extract_features_batch, FEATURE_NAMES

# Attempt XGBoost import with fallback to HistGradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier as XGBClassifier
    HAS_XGBOOST = False

def train_and_benchmark_models(dataset_path=None, sample_size=25000):
    """
    Trains and benchmarks three comparative architectures:
    1. Random Forest (Baseline Ensemble)
    2. XGBoost / Gradient Boosting (State-of-the-art Tree Model)
    3. Deep Learning / Neural Network (MLP / Deep Architecture from IEEE 2024)
    Computes metrics, latency, ROC curves, and SHAP explainability.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # 1. Load Data
    if dataset_path is None:
        if os.path.exists("data/phiusiil_dataset.csv"):
            dataset_path = "data/phiusiil_dataset.csv"
        else:
            dataset_path = "data/sample_dataset.csv"

    print(f"[*] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    if len(df) > sample_size:
        print(f"[i] Using {sample_size:,} balanced samples for benchmarking...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # 2. Load Reduced Features Config if available
    reduced_features_config = "data/reduced_features.json"
    if os.path.exists(reduced_features_config):
        with open(reduced_features_config, "r") as f:
            selected_features = json.load(f).get('reduced_features', FEATURE_NAMES)
        print(f"[+] Loaded {len(selected_features)} reduced features from config: {selected_features}")
    else:
        selected_features = FEATURE_NAMES
        print(f"[i] Using all {len(selected_features)} features.")

    # 3. Extract Features
    has_features = all(c in df.columns for c in selected_features)
    if has_features:
        X = df[selected_features]
    else:
        print("[*] Extracting features from raw URLs...")
        full_X = extract_features_batch(df['url'])
        X = full_X[selected_features]

    y = df['label']

    # 4. Train / Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[+] Train set size: {len(X_train):,}, Test set size: {len(X_test):,}")

    # 5. Define Comparative Models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(random_state=42) if HAS_XGBOOST else XGBClassifier(random_state=42),
        "Deep Neural Net (DL)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    }

    results = []
    roc_data = {}

    print("\n" + "="*70)
    print("TRAINING & BENCHMARKING MODELS (ML vs. DL)")
    print("="*70)

    for name, clf in models.items():
        print(f"\n[*] Training {name}...")
        start_train = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start_train

        # Latency Benchmark: Measure time per 1,000 predictions
        start_pred = time.time()
        y_pred = clf.predict(X_test)
        pred_time = time.time() - start_pred
        latency_ms_per_url = (pred_time / len(X_test)) * 1000

        # Probabilities for ROC
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc,
            "Train_Time_s": train_time,
            "Latency_ms_per_URL": latency_ms_per_url
        })

        print(f"  [+] Accuracy:  {acc*100:.2f}%")
        print(f"  [+] F1-Score:  {f1*100:.2f}%")
        print(f"  [+] ROC-AUC:   {auc:.4f}")
        print(f"  [+] Latency:   {latency_ms_per_url:.4f} ms per URL")

    # Benchmark Summary Table
    results_df = pd.DataFrame(results).set_index("Model")
    print("\n" + "="*70)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("="*70)
    print(results_df.to_string())
    results_df.to_csv("data/model_benchmark_results.csv")
    print("[+] Saved benchmark table to: data/model_benchmark_results.csv")

    # 6. Save Best Model (Random Forest or XGBoost)
    best_model_name = results_df['F1-Score'].idxmax()
    best_model = models[best_model_name]
    print(f"\n[+] Best Model identified: {best_model_name}")

    joblib.dump({
        'model': best_model,
        'model_name': best_model_name,
        'features': selected_features
    }, "models/phishing_detector.pkl")
    print("[+] Saved production model artifact to: models/phishing_detector.pkl")

    # 7. Generate Publication Plot 1: Combined ROC Curves
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance')
    plt.title("ROC Curves: Phishing Detection Models on Reduced URL Features", fontsize=12, pad=12)
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("results_roc_curves.png", dpi=300)
    plt.close()
    print("[+] Saved ROC Curve plot to: results_roc_curves.png")

    # 8. Generate Publication Plot 2: Metrics Comparison Bar Chart
    plt.figure(figsize=(9, 5))
    metrics_to_plot = results_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']] * 100
    metrics_to_plot.plot(kind='bar', figsize=(10, 5), rot=0, colormap='viridis')
    plt.title("Performance Metrics Comparison Across Architectures (%)", fontsize=13, pad=12)
    plt.ylabel("Score (%)", fontsize=11)
    plt.ylim([80, 102])
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("results_metrics_comparison.png", dpi=300)
    plt.close()
    print("[+] Saved Metrics Comparison plot to: results_metrics_comparison.png")

    # 9. SHAP Explainability on Best Tree Model
    print("\n[*] Generating Full Explainable AI (SHAP) Visualizations...")
    explainer_model = models["Random Forest"]
    explainer = shap.TreeExplainer(explainer_model)
    shap_values = explainer.shap_values(X_test.iloc[:500])

    if isinstance(shap_values, list):
        shap_values_class = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_values_class = shap_values[:, :, 1]
    else:
        shap_values_class = shap_values

    # Global SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_class, X_test.iloc[:500], show=False)
    plt.title("Global Explainability: SHAP Feature Impact Summary", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig("shap_summary_full.png", dpi=300)
    plt.close()
    print("[+] Saved Global SHAP plot to: shap_summary_full.png")

    # Save Explainer
    joblib.dump(explainer, "models/shap_explainer.pkl")
    print("[+] Saved SHAP Explainer to: models/shap_explainer.pkl")

    print("\n[✔] Complete training, benchmarking, and XAI pipeline finished successfully!")
    return results_df

if __name__ == "__main__":
    train_and_benchmark_models()
