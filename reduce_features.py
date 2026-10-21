import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from extract_features import extract_features_batch, FEATURE_NAMES

def perform_feature_reduction(dataset_path=None, top_k=10, sample_size=20000):
    """
    Performs systematic feature selection on the 12 lexical URL features using
    Mutual Information (MI) and Random Forest feature importance.
    Evaluates model performance across different feature subset sizes.
    """
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
        print(f"[i] Subsampling to {sample_size:,} rows for rapid feature reduction analysis...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # 2. Extract Features if not pre-extracted
    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
    if len(feature_cols) == len(FEATURE_NAMES):
        print("[+] Dataset already contains extracted feature columns.")
        X = df[FEATURE_NAMES]
    else:
        print("[*] Extracting 12 lexical features from URL column...")
        X = extract_features_batch(df['url'])
        
    y = df['label']

    # 3. Compute Mutual Information Scores
    print("[*] Computing Mutual Information (MI) scores...")
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=FEATURE_NAMES).sort_values(ascending=False)

    # 4. Compute Random Forest Feature Importances
    print("[*] Computing Random Forest Gini Importances...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_series = pd.Series(rf.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)

    # 5. Combined Ranking
    ranking_df = pd.DataFrame({
        'Mutual_Information': mi_series,
        'RF_Importance': rf_series
    })
    ranking_df['Average_Rank'] = (
        ranking_df['Mutual_Information'].rank(ascending=False) + 
        ranking_df['RF_Importance'].rank(ascending=False)
    ) / 2.0
    ranking_df = ranking_df.sort_values(by='Average_Rank')

    print("\n" + "="*60)
    print("FEATURE IMPORTANCE & RANKING SUMMARY")
    print("="*60)
    print(ranking_df.to_string())

    # 6. Evaluate Accuracy and F1 vs. Number of Features (Ablation Study)
    print("\n[*] Evaluating Performance across Feature Subset Sizes (Ablation)...")
    k_values = [2, 4, 6, 8, 10, 12]
    accuracies = []
    f1_scores = []
    
    sorted_features = list(ranking_df.index)

    for k in k_values:
        selected_k = sorted_features[:k]
        rf_k = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf_k.fit(X_train[selected_k], y_train)
        preds = rf_k.predict(X_test[selected_k])
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        accuracies.append(acc)
        f1_scores.append(f1)
        print(f"  - Top {k:2d} Features: Accuracy = {acc*100:.2f}%, F1 = {f1*100:.2f}%")

    # 7. Select Top-K Features
    reduced_features = sorted_features[:top_k]
    print(f"\n[+] Selected Top {top_k} Reduced Features for Production Model:")
    for rank, feat in enumerate(reduced_features, 1):
        print(f"    {rank}. {feat}")

    # Save to JSON configuration
    config = {
        'total_features': len(FEATURE_NAMES),
        'reduced_feature_count': len(reduced_features),
        'reduced_features': reduced_features,
        'ablation_study': {
            'k_values': k_values,
            'accuracy': [round(a, 4) for a in accuracies],
            'f1_score': [round(f, 4) for f in f1_scores]
        }
    }
    with open("data/reduced_features.json", "w") as f:
        json.dump(config, f, indent=4)
    print("[+] Saved configuration to: data/reduced_features.json")

    # 8. Generate Publication Plots
    # Plot 1: Feature Importance Bar Chart
    plt.figure(figsize=(10, 6))
    ranking_df['RF_Importance'].sort_values().plot(kind='barh', color='#2b5c8f')
    plt.title("Random Forest Gini Feature Importance (Lexical URL Features)", fontsize=13, pad=12)
    plt.xlabel("Importance Score", fontsize=11)
    plt.ylabel("Feature Name", fontsize=11)
    plt.tight_layout()
    plt.savefig("feature_importance_ranking.png", dpi=300)
    plt.close()
    print("[+] Saved plot to: feature_importance_ranking.png")

    # Plot 2: Ablation Curve (Accuracy & F1 vs. Number of Features)
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, [a*100 for a in accuracies], marker='o', linewidth=2, label='Accuracy (%)', color='#1f77b4')
    plt.plot(k_values, [f*100 for f in f1_scores], marker='s', linewidth=2, label='F1-Score (%)', color='#2ca02c')
    plt.axvline(x=top_k, color='red', linestyle='--', label=f'Optimal Cutoff (k={top_k})')
    plt.title("Impact of Feature Reduction on Phishing Detection Performance", fontsize=13, pad=12)
    plt.xlabel("Number of Selected Features (k)", fontsize=11)
    plt.ylabel("Performance (%)", fontsize=11)
    plt.xticks(k_values)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("feature_reduction_curve.png", dpi=300)
    plt.close()
    print("[+] Saved plot to: feature_reduction_curve.png")

    return reduced_features

if __name__ == "__main__":
    perform_feature_reduction()
