import re
import urllib.parse
import pandas as pd
from tqdm import tqdm

FEATURE_NAMES = [
    'url_length',
    'hostname_length',
    'path_length',
    'count_dots',
    'count_hyphens',
    'count_queries',
    'count_at',
    'count_slash',
    'has_ip',
    'count_subdomains',
    'has_keyword',
    'has_http_in_host'
]

IP_PATTERN = re.compile(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')
KEYWORDS = ('login', 'secure', 'update', 'verify', 'bank', 'account', 'signin', 'webscr')

def extract_lexical_features(url: str) -> dict:
    """
    Extracts 12 lexical features from a raw URL string.
    Optimized for microsecond-level latency and zero external network calls.
    """
    if not isinstance(url, str):
        url = str(url) if url is not None else ""

    parsed_url = url
    if not url.startswith(('http://', 'https://')):
        parsed_url = 'http://' + url

    try:
        parsed = urllib.parse.urlparse(parsed_url)
        hostname = parsed.netloc or ""
        path = parsed.path or ""
    except Exception:
        hostname = ""
        path = ""

    clean_host = hostname.split(':')[0] if ':' in hostname else hostname
    lower_host = hostname.lower()

    return {
        'url_length': len(url),
        'hostname_length': len(hostname),
        'path_length': len(path),
        'count_dots': url.count('.'),
        'count_hyphens': url.count('-'),
        'count_queries': url.count('?'),
        'count_at': url.count('@'),
        'count_slash': url.count('/'),
        'has_ip': 1 if IP_PATTERN.match(clean_host) else 0,
        'count_subdomains': max(0, hostname.count('.') - 1),
        'has_keyword': 1 if any(kw in lower_host for kw in KEYWORDS) else 0,
        'has_http_in_host': 1 if ('http' in lower_host or 'https' in lower_host) else 0
    }

def extract_features_batch(urls, show_progress=True) -> pd.DataFrame:
    """
    Extracts features for an iterable of URLs with an optional progress bar.
    """
    iterator = tqdm(urls, desc="Extracting URL Features", total=len(urls)) if show_progress else urls
    records = [extract_lexical_features(u) for u in iterator]
    return pd.DataFrame(records, columns=FEATURE_NAMES)

if __name__ == "__main__":
    sample_urls = [
        "https://www.google.com",
        "https://www.paypal.com/signin",
        "http://paypal-security-update-verify.com/login.php",
        "http://192.168.1.100/secure/bank/index.html",
        "https://github.com/features",
        "http://verify-account-webscr.support-login-secure.info/signin"
    ]
    
    print("Extracting lexical features from sample URLs...\n")
    df = extract_features_batch(sample_urls, show_progress=False)
    df.insert(0, 'url', sample_urls)
    print(df.to_string(index=False))
