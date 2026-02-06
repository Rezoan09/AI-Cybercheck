
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from utils import load_csv_safe, ensure_numeric, split_X_y, save_artifacts, normalize_binary_labels

DEF_IDS_LABEL = "Label"

def train_ids(ids_csv: str = None, ids_label: str = DEF_IDS_LABEL, drop_cols: str = ""):
    if ids_csv and os.path.exists(ids_csv):
        df = load_csv_safe(ids_csv)
        drop_list = [c.strip() for c in drop_cols.split(",") if c.strip()]
        if ids_label in df.columns:
            df[ids_label] = normalize_binary_labels(df[ids_label])
        else:
            raise ValueError(f"ids_label '{ids_label}' not found in {df.columns.tolist()}")
        X_num = ensure_numeric(df, drop_list)
        X, y = split_X_y(X_num, ids_label)
    else:
        rng = np.random.RandomState(42)
        X = pd.DataFrame({
            "Flow_Duration": rng.exponential(scale=2000, size=1000),
            "Total_Fwd_Packets": rng.poisson(lam=10, size=1000),
            "Total_Bwd_Packets": rng.poisson(lam=9, size=1000),
            "Fwd_Packet_Length_Mean": rng.normal(400, 100, size=1000).clip(0, None),
            "Bwd_Packet_Length_Mean": rng.normal(380, 110, size=1000).clip(0, None),
            "Flow_Bytes_s": rng.normal(10000, 3000, size=1000).clip(0, None),
        })
        y = pd.Series((X["Flow_Bytes_s"] > 11000).astype(int), name=ids_label)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight = {c: w for c, w in zip(classes, weights)}

    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=13, class_weight=class_weight)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("\n=== IDS Model Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    save_artifacts({"ids_model": clf, "ids_columns": X.columns.tolist()}, dirpath="models")
    print("Saved IDS model to models/")

def train_phish(phish_csv: str = None, text_col: str = "message", label_col: str = "label"):
    if phish_csv and os.path.exists(phish_csv):
        df = load_csv_safe(phish_csv)
        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"Expected columns '{text_col}' and '{label_col}'. Found: {df.columns.tolist()}")
        X_text = df[text_col].astype(str)
        y = normalize_binary_labels(df[label_col])
    else:
        data = [
            ("URGENT: Your account will be closed, click this link", 1),
            ("Meeting rescheduled to 3pm, see you then", 0),
            ("Verify your password immediately", 1),
            ("Invoice attached for last month", 0),
            ("Win a free iPhone now!!!", 1),
            ("Project update: repository merged successfully", 0),
        ]
        X_text = pd.Series([t for t, _ in data], name=text_col)
        y = pd.Series([l for _, l in data], name=label_col)

    X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=11, stratify=y)
    pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))), ("clf", LogisticRegression(max_iter=200))])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("\n=== Phishing Model Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    save_artifacts({"phish_model": pipe}, dirpath="models")
    print("Saved phishing model to models/")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids_csv", type=str, default=None)
    parser.add_argument("--ids_label", type=str, default=DEF_IDS_LABEL)
    parser.add_argument("--drop_cols", type=str, default="Flow_ID, Timestamp, Src_IP, Src_Port, Dst_IP, Dst_Port, Protocol")
    parser.add_argument("--phish_csv", type=str, default=None)
    parser.add_argument("--phish_text_col", type=str, default="message")
    parser.add_argument("--phish_label_col", type=str, default="label")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    train_ids(ids_csv=args.ids_csv, ids_label=args.ids_label, drop_cols=args.drop_cols)
    train_phish(phish_csv=args.phish_csv, text_col=args.phish_text_col, label_col=args.phish_label_col)

if __name__ == "__main__":
    main()
