"""
Protocol guards (U3.1, U3.5) and the new M2 feature groups (U2.2, U2.3).
Run on M1's real frozen splits.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset  # noqa: E402
from features import build_feature_transformer  # noqa: E402
from upstream_preprocess import CleanTransformer  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return dataset.load_typed()


@pytest.fixture(scope="module")
def splits():
    return dataset.load_splits()


def test_holdout_is_locked_by_default(df, splits):
    with pytest.raises(dataset.HoldoutLockedError):
        dataset.holdout_frame(df, splits)


def test_no_fold_touches_the_holdout(df, splits):
    holdout = set(splits["holdout_ids"])
    for k in range(len(splits["folds"])):
        X_tr, _, X_va, _ = dataset.fold_frames(df, splits, k)
        assert holdout.isdisjoint(X_tr["id"])
        assert holdout.isdisjoint(X_va["id"])
        assert set(X_tr["id"]).isdisjoint(X_va["id"])


def test_assert_no_holdout_catches_a_leak(splits):
    with pytest.raises(AssertionError):
        dataset.assert_no_holdout(splits["dev_ids"][:5] + splits["holdout_ids"][:1], splits)


def test_validation_folds_partition_the_dev_set(splits):
    val = [i for f in splits["folds"] for i in f["val_ids"]]
    assert sorted(val) == sorted(splits["dev_ids"])


def test_fold_prevalence_within_half_a_point(df, splits):
    """Proposal slide 8 acceptance criterion."""
    overall = df["stroke"].mean()
    for k in range(len(splits["folds"])):
        _, _, _, y_va = dataset.fold_frames(df, splits, k)
        assert abs(y_va.mean() - overall) < 0.005


def _fit(df, splits, **cfg):
    X_tr, _, X_va, _ = dataset.fold_frames(df, splits, 0)
    clean = CleanTransformer().fit(X_tr)
    feat = build_feature_transformer(**cfg).fit(clean.transform(X_tr))
    return feat.transform(clean.transform(X_va)), X_va


def test_raw_config_has_no_derived_columns(df, splits):
    out, _ = _fit(df, splits, bands=False, glucose_bands=False, comorbidity=False)
    derived = ("age_band", "bmi_category", "glucose_band", "comorbidity_count", "age_x_hypertension")
    assert not [c for c in out.columns if c.startswith(derived)]


def test_glucose_bands_use_documented_cutoffs(df, splits):
    out, X_va = _fit(df, splits, bands=False, glucose_bands=True)
    g = X_va["avg_glucose_level"].values
    assert (out["glucose_band_very_high"].values == (g >= 200)).all()
    assert (out["glucose_band_normal"].values == (g < 100)).all()
    assert (out.filter(like="glucose_band_").sum(axis=1) == 1).all()


def test_comorbidity_features_are_correct_before_scaling(df, splits):
    from features import BandFeatures

    _, _, X_va, _ = dataset.fold_frames(df, splits, 0)
    X = CleanTransformer().fit(X_va).transform(X_va)
    out = BandFeatures(bands=False, comorbidity=True).transform(X)
    assert (out["comorbidity_count"] == X["hypertension"] + X["heart_disease"]).all()
    assert np.allclose(out["age_x_hypertension"], X["age"] * X["hypertension"])


def test_all_groups_on_still_gives_zero_nans_and_stable_columns(df, splits):
    out, _ = _fit(df, splits, bands=True, glucose_bands=True, comorbidity=True)
    assert out.isna().sum().sum() == 0
    out2, _ = _fit(df, splits, bands=True, glucose_bands=True, comorbidity=True)
    assert list(out.columns) == list(out2.columns)


def test_saved_model_was_never_fit_on_holdout_ids(splits):
    """The final model's BMI median must equal the dev-set median, which it
    only can if the holdout was excluded from the refit."""
    path = dataset.ROOT / "models" / "final_model.joblib"
    if not path.exists():
        pytest.skip("no trained model yet; run train.py")
    import joblib

    model = joblib.load(path)
    df = dataset.load_typed()
    X_dev, _ = dataset.dev_frame(df, splits)
    fill = model.pipeline.named_steps["clean"].bmi_fill_value_
    assert fill == pytest.approx(X_dev["bmi"].median())
