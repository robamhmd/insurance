"""Shared feature engineering, used by train.py AND app.py so both always match."""
import pandas as pd

NUMERIC = ["age", "bmi", "children", "obese", "smoker_obese"]
BINARY = ["sex", "smoker"]
CATEGORICAL = ["region"]
FEATURES = NUMERIC + BINARY + CATEGORICAL


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix inconsistent text entries (stateless, safe to run before the split)."""
    df = df.copy()
    for col in ["sex", "smoker", "region"]:
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature engineering (uses no statistics, so no leakage risk)."""
    df = clean(df)
    df["sex"] = (df["sex"] == "male").astype(int)
    df["smoker"] = (df["smoker"] == "yes").astype(int)
    df["obese"] = (df["bmi"] >= 30).astype(int)
    df["smoker_obese"] = df["smoker"] * df["obese"]
    return df[FEATURES]
