import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import shap

# Import our feature extractor
from extract_features import extract_lexical_features

def generate_synthetic_data(num_samples=1000):
    """
    Generates a realistic synthetic dataset of benign and phishing URLs 
    to enable a working baseline demonstration immediately.
    """
    np.random.seed(42)
    urls = []
    labels = []
    
    # Common patterns for generation
    benign_domains = ["google.com", "yahoo.com", "github.com", "wikipedia.org", "nytimes.com", "microsoft.com"]
    phish_keywords = ["secure-login", "verify-bank", "update-paypal", "signin-security", "webscr-update"]
    
    for i in range(num_samples):
        is_phishing = np.random.choice([0, 1])
        labels.append(is_phishing)
        
        if is_phishing == 0:
            # Benign URL
            domain = np.random.choice(benign_domains)
            path = np.random.choice(["", "/index.html", "/features", "/about/us", "/search?q=test"])
            url = f"https://www.{domain}{path}"
        else:
            # Phishing URL
            keyword = np.random.choice(phish_keywords)
            domain_extension = np.random.choice(["info", "cc", "top", "xyz", "support-update.net"])
            path = np.random.choice(["/login.php", "/signin/verify", "/bank/account.html", "/webscr?cmd=_login-run"])
            
            # Often contain IP addresses or query strings or extra subdomains
            use_ip = np.random.choice([True, False], p=[0.2, 0.8])
            if use_ip:
                ip_addr = f"{np.random.randint(100, 200)}.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}.{np.random.randint(1, 254)}"
                url = f"http://{ip_addr}{path}"
            else:
                subdomain = "secure-update" if np.random.choice([True, False]) else "verify"
                url = f"http://{subdomain}.{keyword}.{domain_extension}{path}"
                
        urls.append(url)
        
    return pd.DataFrame({"url": urls, "label": labels})

def main():
    print("Step 1: Loading Dataset...")
    # Generate synthetic data for demonstration, or load custom CSV if available
    csv_path = "data/phishing_urls.csv"
    if os.path.exists(csv_path):
        print(f"Loading custom dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
        print("No custom dataset found. Generating synthetic dataset for demonstration...")
        df = generate_synthetic_data(1000)
        # Create data folder and save synthetic dataset for user inspection
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/sample_dataset.csv", index=False)
        print("Saved sample dataset to data/sample_dataset.csv")

    print("\nStep 2: Extracting Features...")
    feature_list = []
    for url in df['url']:
        feature_list.append(extract_lexical_features(url))
        
    X = pd.DataFrame(feature_list)
    y = df['label']
    
    print(f"Extracted {X.shape[1]} features for {X.shape[0]} URLs.")
    print("Features extracted:", list(X.columns))

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nStep 3: Training Random Forest Classifier (Baseline Model)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nStep 4: Generating Explainable AI (SHAP) Interpretation...")
    # Use TreeExplainer for Random Forest
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    
    # Note: shap_values for binary classification in TreeExplainer can be a list of two arrays, 
    # a 3D numpy array (num_samples, num_features, 2), or a 2D numpy array (num_samples, num_features).
    if isinstance(shap_values, list):
        shap_values_class = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_values_class = shap_values[:, :, 1]
    else:
        shap_values_class = shap_values

    # Plot 1: Feature Importance Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_class, X_test, show=False)
    plt.title("SHAP Feature Importance (Explainable AI Summary)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=300)
    plt.close()
    print("Saved SHAP summary plot to: shap_summary.png")

    # Plot 2: Local Explanation for a single Phishing URL prediction
    # Find a phishing instance in the test set
    phish_indices = np.where(y_test == 1)[0]
    if len(phish_indices) > 0:
        idx = phish_indices[0]
        plt.figure(figsize=(10, 4))
        # Use waterfall or force plot representation via bar chart for easy reading
        shap.bar_plot(shap_values_class[idx], max_display=10, feature_names=X_test.columns, show=False)
        plt.title(f"SHAP Local Explanation for Phishing URL Sample\nURL: {df.iloc[X_test.index[idx]]['url'][:60]}...", fontsize=10, pad=15)
        plt.tight_layout()
        plt.savefig("shap_local_explanation.png", dpi=300)
        plt.close()
        print("Saved local SHAP explanation plot to: shap_local_explanation.png")

    print("\nBaseline demonstration ran successfully! You can present these metrics and plots for your 10-day assessment.")

if __name__ == "__main__":
    main()
