import os
import sys
import joblib
import pandas as pd
import numpy as np

from extract_features import extract_lexical_features, FEATURE_NAMES

def predict_single_url(url: str, model_path="models/phishing_detector.pkl", explainer_path="models/shap_explainer.pkl"):
    """
    Predicts whether a single URL is Legitimate or Phishing, and provides
    instant, human-readable explanations of the decision.
    """
    if not os.path.exists(model_path):
        print(f"[!] Model not found at '{model_path}'. Please run 'python train_full_models.py' first.")
        return

    # Load model package
    artifact = joblib.load(model_path)
    model = artifact['model']
    selected_features = artifact.get('features', FEATURE_NAMES)

    # Extract features
    raw_features = extract_lexical_features(url)
    df_features = pd.DataFrame([raw_features])[selected_features]

    # Predict
    prob = model.predict_proba(df_features)[0, 1] if hasattr(model, "predict_proba") else 0.5
    is_phishing = prob >= 0.5

    status_icon = "🚨 PHISHING DETECTED" if is_phishing else "✅ LEGITIMATE (SAFE)"
    confidence = prob if is_phishing else (1.0 - prob)

    print("\n" + "="*65)
    print("PHISHING URL DETECTION & EXPLAINABILITY REPORT")
    print("="*65)
    print(f"Target URL   : {url}")
    print(f"Verdict      : {status_icon}")
    print(f"Confidence   : {confidence * 100:.2f}%")
    print(f"Phish Risk   : {prob * 100:.2f}%")
    print("-" * 65)
    print("EXTRACTED REDUCED LEXICAL FEATURES:")
    for feat in selected_features:
        print(f"  • {feat:<20}: {raw_features[feat]}")

    # Explainability Breakdown
    if os.path.exists(explainer_path):
        try:
            explainer = joblib.load(explainer_path)
            shap_values = explainer.shap_values(df_features)

            if isinstance(shap_values, list):
                shap_class = shap_values[1][0]
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                shap_class = shap_values[0, :, 1]
            else:
                shap_class = shap_values[0]

            feature_impact = pd.Series(shap_class, index=selected_features).sort_values(ascending=False)
            
            print("-" * 65)
            print("EXPLAINABLE AI (SHAP) DECISION FACTORS:")
            print("(Positive pushes towards PHISHING, Negative pulls towards SAFE)")
            for feat, impact in feature_impact.items():
                direction = "🔺 Increased Risk" if impact > 0 else "🔻 Decreased Risk"
                print(f"  • {feat:<20}: {impact:+.4f} ({direction})")
        except Exception as e:
            pass

    print("="*65 + "\n")
    return {"url": url, "is_phishing": bool(is_phishing), "confidence": float(confidence)}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_url = sys.argv[1]
    else:
        test_url = input("Enter a URL to test (e.g. http://paypal-verify-login.xyz/update): ").strip()
        if not test_url:
            test_url = "http://paypal-security-update-verify.com/login.php"

    predict_single_url(test_url)
