
import os
import joblib
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

NUMERIC_FILL_VALUE = 0.0

def load_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    return df

def ensure_numeric(df: pd.DataFrame, drop_cols: Optional[List[str]] = None) -> pd.DataFrame:
    drop_cols = drop_cols or []
    df2 = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()
    for c in df2.columns:
        if df2[c].dtype == 'O':
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
    df2 = df2.fillna(NUMERIC_FILL_VALUE)
    return df2

def split_X_y(df: pd.DataFrame, label_col: str):
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe columns: {df.columns.tolist()}")
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y

def save_artifacts(artifacts: dict, dirpath: str = "."):
    os.makedirs(dirpath, exist_ok=True)
    for name, obj in artifacts.items():
        joblib.dump(obj, os.path.join(dirpath, f"{name}.pkl"))

def load_artifact(name: str, dirpath: str = "."):
    path = os.path.join(dirpath, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)

def normalize_binary_labels(series: pd.Series) -> pd.Series:
    def to01(v):
        if isinstance(v, str):
            lv = v.strip().lower()
            if lv in ["0", "benign", "normal", "ham", "legitimate", "legit", "safe"]:
                return 0
            else:
                return 1
        try:
            return 0 if int(v) == 0 else 1
        except Exception:
            return 1
    return series.apply(to01)
