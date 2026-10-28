"""
U2.5 -- Leakage audit for the M2 FeatureTransformer, run against Jyot's real
data (stroke_typed.parquet, splits.json) and his real CleanTransformer.

Claim under test: fitting on the training fold and transforming the
validation fold never uses statistics computed from the validation fold.

Run with: python -m pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features import build_feature_transformer  # noqa: E402
from upstream_preprocess import CleanTransformer  # noqa: E402  (M1's deliverable)


@pytest.fixture(scope="module")
def real_df():
    return pd.read_parquet(ROOT / "data" / "stroke_typed.parquet")


@pytest.fixture(scope="module")
def real_splits():
    return json.load(open(ROOT / "data" / "splits_REAL.json"))


@pytest.fixture
def fold0(real_df, real_splits):
    """M1's frozen fold 0 -- the folds every member must share."""
    f = real_splits["folds"][0]
    train_df = real_df[real_df["id"].isin(f["train_ids"])].drop(columns=["stroke"])
    val_df = real_df[real_df["id"].isin(f["val_ids"])].drop(columns=["stroke"])
    assert set(train_df["id"]).isdisjoint(set(val_df["id"])), "frozen folds overlap"
    return train_df, val_df


def _fit_full_pipeline(train_df):
    """M1 CleanTransformer -> M2 FeatureTransformer, both fit on train only."""
    cleaner = CleanTransformer()
    train_clean = cleaner.fit_transform(train_df)
    feat = build_feature_transformer()
    feat.fit(train_clean)
    return cleaner, feat


def test_fit_train_transform_val_does_not_raise(fold0):
    train_df, val_df = fold0
    cleaner, feat = _fit_full_pipeline(train_df)
    out = feat.transform(cleaner.transform(val_df))
    assert out.shape[0] == val_df.shape[0]


def test_bmi_imputation_uses_only_training_median(fold0):
    train_df, _ = fold0
    cleaner, _ = _fit_full_pipeline(train_df)
    assert cleaner.bmi_fill_value_ == pytest.approx(train_df["bmi"].median(), abs=1e-9)


def test_transforming_poisoned_validation_data_does_not_refit(fold0):
    """Poison the val fold with extreme values and push it through the fitted
    pipeline. If transform() secretly refit anything, the BMI median or the
    scaler means would move. They must stay exactly at their train-fold values."""
    train_df, val_df = fold0
    cleaner, feat = _fit_full_pipeline(train_df)
    scaler = feat.named_steps["encode_scale"].named_transformers_["numeric"]
    median_before, means_before = cleaner.bmi_fill_value_, scaler.mean_.copy()

    poisoned_val = val_df.copy()
    poisoned_val["bmi"] = 999.0
    poisoned_val["age"] = 999.0
    feat.transform(cleaner.transform(poisoned_val))

    assert cleaner.bmi_fill_value_ == pytest.approx(median_before, abs=1e-12)
    assert scaler.mean_ == pytest.approx(means_before, abs=1e-12)
    assert cleaner.bmi_fill_value_ < 900


def test_scaler_statistics_come_only_from_training_fold(fold0):
    train_df, val_df = fold0
    _, feat = _fit_full_pipeline(train_df)

    scaler = feat.named_steps["encode_scale"].named_transformers_["numeric"]
    fitted_mean_age = scaler.mean_[0]  # age is first in NUMERIC_COLUMNS

    combined_mean_age = pd.concat([train_df["age"], val_df["age"]]).mean()
    assert fitted_mean_age == pytest.approx(train_df["age"].mean(), abs=1e-6)
    assert fitted_mean_age != pytest.approx(combined_mean_age, abs=1e-3)


def test_output_has_fixed_column_order(fold0):
    train_df, val_df = fold0
    cleaner, feat = _fit_full_pipeline(train_df)
    out_train = feat.transform(cleaner.transform(train_df))
    out_val = feat.transform(cleaner.transform(val_df))
    assert list(out_train.columns) == list(out_val.columns)


def test_rare_gender_other_does_not_crash(real_df, real_splits):
    """gender='Other' has n=1. Whichever fold it lands in, transform must not raise."""
    f = real_splits["folds"][0]
    train_df = real_df[real_df["id"].isin(f["train_ids"])].drop(columns=["stroke"])
    val_df = real_df[real_df["id"].isin(f["val_ids"])].drop(columns=["stroke"])

    cleaner, feat = _fit_full_pipeline(train_df)
    out = feat.transform(cleaner.transform(val_df))
    assert out.shape[0] == val_df.shape[0]


def test_zero_nans_after_full_handoff(fold0):
    train_df, val_df = fold0
    cleaner, feat = _fit_full_pipeline(train_df)
    out = feat.transform(cleaner.transform(val_df))
    assert out.isna().sum().sum() == 0
