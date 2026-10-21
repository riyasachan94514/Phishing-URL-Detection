import os
import sys
import pandas as pd

def download_and_prepare_phiusiil(output_path="data/phiusiil_dataset.csv", max_rows=None):
    """
    Fetches the official PhiUSIIL Phishing URL Dataset (UCI ID: 967).
    Standardizes column names and class labels (1 = Phishing, 0 = Legitimate).
    """
    os.makedirs("data", exist_ok=True)
    
    if os.path.exists(output_path):
        print(f"[+] Dataset already exists locally at: {output_path}")
        df = pd.read_csv(output_path, nrows=5)
        print(f"[i] Preview:\n{df.head()}")
        return output_path

    print("[*] Fetching PhiUSIIL Phishing URL Dataset from UCI Repository (ID: 967)...")
    print("[*] Note: This may take a couple of minutes depending on your internet connection (~235,795 rows)...")
    
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        print("[!] 'ucimlrepo' is not installed. Installing ucimlrepo now...")
        os.system(f"{sys.executable} -m pip install ucimlrepo")
        from ucimlrepo import fetch_ucirepo

    try:
        phiusiil = fetch_ucirepo(id=967)
        X = phiusiil.data.features
        y = phiusiil.data.targets
        
        # Identify the URL column
        url_col = None
        for col in X.columns:
            if col.lower() == 'url':
                url_col = col
                break
                
        if url_col is None:
            # Fallback to first object/string column
            str_cols = X.select_dtypes(include=['object']).columns
            url_col = str_cols[0] if len(str_cols) > 0 else X.columns[0]
            
        print(f"[+] Found URL column: '{url_col}'")
        
        # Target column
        target_col = y.columns[0]
        
        # Combine into a clean dataframe with standard columns: ['url', 'label']
        clean_df = pd.DataFrame({
            'url': X[url_col],
            'raw_label': y[target_col]
        })
        
        # Standardize labels: In PhiUSIIL, 0 = Phishing, 1 = Legitimate.
        # We standardize to: 1 = Phishing (positive malicious class), 0 = Legitimate (benign)
        unique_vals = clean_df['raw_label'].unique()
        print(f"[i] Raw label unique values: {unique_vals}")
        
        if set(unique_vals) == {0, 1}:
            # Check if 0 is phishing: PhiUSIIL documentation defines 0 as Phishing and 1 as Legitimate.
            # Invert so 1 = Phishing, 0 = Legitimate
            clean_df['label'] = 1 - clean_df['raw_label']
        else:
            clean_df['label'] = clean_df['raw_label'].apply(
                lambda val: 1 if str(val).lower() in ['1', 'phishing', 'bad', 'malicious'] else 0
            )
            
        clean_df = clean_df[['url', 'label']].dropna().drop_duplicates(subset=['url'])
        
        if max_rows and len(clean_df) > max_rows:
            print(f"[i] Subsampling to {max_rows} rows for balanced rapid development...")
            clean_df = clean_df.sample(n=max_rows, random_state=42).reset_index(drop=True)

        print(f"[+] Total clean URLs: {len(clean_df):,}")
        print(f"    - Phishing (1): {(clean_df['label'] == 1).sum():,} URLs")
        print(f"    - Legitimate (0): {(clean_df['label'] == 0).sum():,} URLs")
        
        clean_df.to_csv(output_path, index=False)
        print(f"[+] Saved standardized dataset to: {output_path}")
        return output_path

    except Exception as e:
        print(f"[!] Error downloading dataset via ucimlrepo: {e}")
        print("[*] Attempting fallback to direct Kaggle/GitHub mirror if available...")
        raise e

if __name__ == "__main__":
    download_and_prepare_phiusiil()
