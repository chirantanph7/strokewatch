"""
StrokeWatch -- M2 Feature Engineering (Chirantan)

Implements:
  U2.1  Age bands and WHO-adult BMI categories
  U2.2  Glucose bands (provisional, fasting status unknown; non-diagnostic)
  U2.3  Comorbidity count and age x hypertension interaction
  U2.4  Encoding of nominal columns + scaling of numeric columns

Each derived group can be switched on or off, so the U2.7 ablation
(train.py) can compare raw against engineered features on identical folds.
Only the groups the ablation supports go into the final model.

Design contract (confirmed against Jyot's M1 handoff -- stroke_typed.parquet,
schema.yaml, splits.json, and preprocess.CleanTransformer):
  - Fit ONLY inside training folds, never on the full dataset.
  - Input is M1's CleanTransformer output: the 10 raw predictors plus
    `bmi_missing`. `id` and `_flags` are identifiers/audit trail and are
    dropped here, not by M1, so the audit trail survives as long as possible.
  - `gender` has a real "Other" category (n=1). handle_unknown="ignore" lets a
    fold that never saw it at fit time pass it through without crashing.
  - Output is a numeric matrix with fixed column order and readable names
    (get_feature_names_out), used by the models and for SHAP labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NOMINAL_COLUMNS = ["gender", "work_type", "Residence_type", "smoking_status", "ever_married"]
NUMERIC_COLUMNS = ["age", "avg_glucose_level", "bmi"]
BINARY_COLUMNS = ["hypertension", "heart_disease", "bmi_missing"]
NON_FEATURE_COLUMNS = ["id", "_flags"]  # identifiers / audit trail

# WHO adult BMI cut-offs (U2.1)
BMI_BINS = [-np.inf, 18.5, 25, 30, np.inf]
BMI_LABELS = ["underweight", "normal", "overweight", "obese"]

# Coarse age bands, chosen for readable SHAP groups
AGE_BINS = [-np.inf, 18, 40, 60, np.inf]
AGE_LABELS = ["child", "adult", "middle_age", "senior"]

# Glucose bands (U2.2). The dataset does not say whether readings are fasting
# or random, so these are heuristic and labelled non-diagnostic: 140 mg/dL is
# the 2-hour OGTT impaired-tolerance cut-off, 200 the random-glucose diabetes one.
GLUCOSE_BINS = [-np.inf, 100, 140, 200, np.inf]
GLUCOSE_LABELS = ["normal", "elevated", "high", "very_high"]

# Optional groups the ablation switches on and off (U2.7)
FEATURE_GROUPS = ["bands", "glucose_bands", "comorbidity"]


def _cut(values: pd.Series, bins, labels) -> pd.Series:
    # astype(str) on a Categorical keeps missing values as real NaN, so
    # fillna is needed; the string check covers older pandas versions.
    out = pd.cut(values, bins=bins, labels=labels).astype(str).fillna("missing")
    return out.mask(out == "nan", "missing")


class BandFeatures(BaseEstimator, TransformerMixin):
    """Adds the derived columns that are switched on and drops id/_flags.
    Stateless: bin edges are fixed clinical cut-offs, not fit from data.

      bands          age_band, bmi_category (U2.1)
      glucose_bands  glucose_band (U2.2)
      comorbidity    comorbidity_count, age_x_hypertension (U2.3)
    """

    def __init__(self, bands: bool = True, glucose_bands: bool = False, comorbidity: bool = False):
        self.bands = bands
        self.glucose_bands = glucose_bands
        self.comorbidity = comorbidity

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in X.columns])
        if self.bands:
            X["age_band"] = _cut(X["age"], AGE_BINS, AGE_LABELS)
            X["bmi_category"] = _cut(X["bmi"], BMI_BINS, BMI_LABELS)
        if self.glucose_bands:
            X["glucose_band"] = _cut(X["avg_glucose_level"], GLUCOSE_BINS, GLUCOSE_LABELS)
        if self.comorbidity:
            X["comorbidity_count"] = X["hypertension"].astype(int) + X["heart_disease"].astype(int)
            X["age_x_hypertension"] = X["age"] * X["hypertension"]
        return X

    def derived_nominal(self) -> list[str]:
        cols = ["age_band", "bmi_category"] if self.bands else []
        return cols + (["glucose_band"] if self.glucose_bands else [])

    def derived_numeric(self) -> list[str]:
        return ["comorbidity_count", "age_x_hypertension"] if self.comorbidity else []

    def get_feature_names_out(self, input_features=None):
        kept = [c for c in input_features if c not in NON_FEATURE_COLUMNS]
        return np.array(kept + self.derived_nominal() + self.derived_numeric())


def build_feature_transformer(
    bands: bool = True, glucose_bands: bool = False, comorbidity: bool = False
) -> Pipeline:
    """Band features -> encode/scale. Fit this ONLY on a training fold.
    With every group off this is the "raw" feature set used by the ablation."""
    band_step = BandFeatures(bands=bands, glucose_bands=glucose_bands, comorbidity=comorbidity)

    encode_scale = ColumnTransformer(
        transformers=[
            (
                "nominal",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                NOMINAL_COLUMNS + band_step.derived_nominal(),
            ),
            ("numeric", StandardScaler(), NUMERIC_COLUMNS + band_step.derived_numeric()),
            ("binary", FunctionTransformer(validate=False), BINARY_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    encode_scale.set_output(transform="pandas")

    return Pipeline(steps=[("bands", band_step), ("encode_scale", encode_scale)])


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from upstream_preprocess import CleanTransformer  # M1's deliverable

    df = pd.read_parquet("data/stroke_typed.parquet")
    splits = json.load(open("data/splits_REAL.json"))

    fold0 = splits["folds"][0]
    train_df = df[df["id"].isin(fold0["train_ids"])].drop(columns=["stroke"])
    val_df = df[df["id"].isin(fold0["val_ids"])].drop(columns=["stroke"])

    # M1 -> M2 handoff: both fit on the training fold only.
    cleaner = CleanTransformer()
    train_clean = cleaner.fit_transform(train_df)
    val_clean = cleaner.transform(val_df)

    pipe = build_feature_transformer()
    pipe.fit(train_clean)
    train_out = pipe.transform(train_clean)
    val_out = pipe.transform(val_clean)

    print(f"Real data: {df.shape[0]} rows total, fold 0: {len(train_df)} train / {len(val_df)} val")
    print(f"Train output shape: {train_out.shape}")
    print(f"Val output shape:   {val_out.shape}")
    print(f"NaNs in train output: {train_out.isna().sum().sum()}")
    print(f"NaNs in val output:   {val_out.isna().sum().sum()}")
    print(f"Output columns ({len(train_out.columns)}):")
    for c in train_out.columns:
        print(f"  - {c}")
