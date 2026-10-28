"""
preprocess.py — StrokeWatch M1 deliverable (Jyot)

Produces:
  1. load_and_type(raw_csv_path) -> typed, de-duplicated DataFrame (no leakage risk, this is row-level only)
  2. CleanTransformer -> sklearn-compatible transformer for the ONE step that must be
     fit only on training folds: BMI median imputation + missing-indicator.

Everything else in this file (typing, range flags, duplicate checks) is row-level and
safe to run once on the whole raw file, because it uses no statistics derived from
the target or from the full-dataset distribution.

Usage:
    df = load_and_type("healthcare-dataset-stroke-data.csv")
    # ... split into train/test using make_splits.py ...
    ct = CleanTransformer()
    train_clean = ct.fit_transform(train_df)
    test_clean  = ct.transform(test_df)   # uses train-fold median, no leakage
"""

from __future__ import annotations
import hashlib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

RAW_DTYPES = {
    "id": "int64",
    "gender": "category",
    "age": "float64",
    "hypertension": "int64",
    "heart_disease": "int64",
    "ever_married": "category",
    "work_type": "category",
    "Residence_type": "category",
    "avg_glucose_level": "float64",
    "bmi": "float64",
    "smoking_status": "category",
    "stroke": "int64",
}

PLAUSIBLE_RANGES = {
    "age": (0.0, 100.0),
    "avg_glucose_level": (40.0, 300.0),
    "bmi": (10.0, 80.0),
}


def file_checksum(path: str) -> str:
    """SHA-256 of the raw file — record this in LICENCE_NOTE.md so re-downloads are verifiable."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_type(raw_csv_path: str) -> pd.DataFrame:
    """
    U1.1 / U1.2 / U1.5: load, type, and flag (never silently drop) anomalies.
    Returns a DataFrame with correct dtypes and an added `_flags` column listing
    any row-level issues found, so nothing is dropped without a paper trail.
    """
    df = pd.read_csv(raw_csv_path, na_values=["N/A", "Unknown_placeholder_never_used"])

    flags = pd.Series([[] for _ in range(len(df))], index=df.index)

    # duplicates
    dup_ids = df["id"].duplicated(keep=False)
    if dup_ids.any():
        flags[dup_ids] = flags[dup_ids].apply(lambda x: x + ["duplicate_id"])

    full_dup = df.duplicated(keep=False)
    if full_dup.any():
        flags[full_dup] = flags[full_dup].apply(lambda x: x + ["duplicate_row"])

    # dtype casting
    for col, dtype in RAW_DTYPES.items():
        if col == "bmi":
            df[col] = pd.to_numeric(df[col], errors="coerce")  # keeps NaN for missing BMI
        elif dtype == "category":
            df[col] = df[col].astype("category")
        else:
            df[col] = df[col].astype(dtype)

    # plausible-range flags (flag, do not drop)
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        out_of_range = ~df[col].between(lo, hi) & df[col].notna()
        if out_of_range.any():
            flags[out_of_range] = flags[out_of_range].apply(lambda x: x + [f"{col}_out_of_range"])

    # known anomaly: gender == 'Other'
    other_gender = df["gender"] == "Other"
    if other_gender.any():
        flags[other_gender] = flags[other_gender].apply(lambda x: x + ["gender_other"])

    # sanity: work_type == 'children' with implausibly high age
    child_adult_mismatch = (df["work_type"] == "children") & (df["age"] > 18)
    if child_adult_mismatch.any():
        flags[child_adult_mismatch] = flags[child_adult_mismatch].apply(
            lambda x: x + ["work_type_age_mismatch"]
        )

    df["_flags"] = flags.apply(lambda x: ";".join(x) if x else "")
    return df


class CleanTransformer(BaseEstimator, TransformerMixin):
    """
    U1.3: the ONE statistic-dependent step — BMI imputation.
    Fit on training folds only. Adds `bmi_missing` indicator before imputing,
    so "missingness" itself stays available as a signal for M2/M3.
    """

    def __init__(self, strategy: str = "median"):
        self.strategy = strategy

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        if self.strategy == "median":
            self.bmi_fill_value_ = X["bmi"].median()
        elif self.strategy == "mean":
            self.bmi_fill_value_ = X["bmi"].mean()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["bmi_missing"] = X["bmi"].isna().astype(int)
        X["bmi"] = X["bmi"].fillna(self.bmi_fill_value_)
        return X


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py <path_to_raw_csv>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Checksum (record this in LICENCE_NOTE.md): {file_checksum(path)}")
    df = load_and_type(path)
    n_flagged = (df["_flags"] != "").sum()
    print(f"Loaded {len(df)} rows, {df['stroke'].sum()} positive, {n_flagged} rows flagged.")
    out_path = "stroke_typed.parquet"
    df.to_parquet(out_path)
    print(f"Wrote {out_path}")
