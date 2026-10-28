"""
U5 checks: SHAP additivity, feature-name mapping, upload validation, and a
dashboard smoke test (proposal slide 24).
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
import dataset  # noqa: E402
from explain import Explainer, readable, source_variable  # noqa: E402
from validation import validate_records  # noqa: E402

MODEL = ROOT / "models" / "final_model.joblib"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="no trained model yet; run train.py")
SAMPLE = ROOT / "app" / "sample_records.csv"


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL)


@pytest.fixture(scope="module")
def dev():
    X, y = dataset.dev_frame(dataset.load_typed(), dataset.load_splits())
    return X


@needs_model
def test_shap_additivity(model, dev):
    """base value + sum of SHAP values = the model's log-odds, per record."""
    e = Explainer(model, dev)
    X = dev.iloc[:100]
    contrib = e.shap_columns(X).sum(axis=1).values
    margin = e.margin(model.transform_features(X))
    assert np.allclose(contrib + e.base_value, margin, atol=1e-4)


@needs_model
def test_grouping_to_source_variables_preserves_the_total(model, dev):
    e = Explainer(model, dev)
    X = dev.iloc[:50]
    assert np.allclose(e.shap_columns(X).sum(axis=1), e.shap_sources(X).sum(axis=1))


@needs_model
def test_every_feature_column_maps_to_a_readable_source(model):
    for col in model.feature_names:
        assert readable(source_variable(col))


@needs_model
def test_feature_names_json_matches_model(model):
    import json

    assert json.loads((ROOT / "feature_names.json").read_text()) == model.feature_names


def _sample():
    return pd.read_csv(SAMPLE, dtype=str, keep_default_na=False)


def test_sample_records_pass_validation():
    records, errors, _ = validate_records(_sample())
    assert errors == [] and len(records) == 6


def test_missing_column_gives_a_readable_error():
    records, errors, _ = validate_records(_sample().drop(columns=["age"]))
    assert records is None and "age" in errors[0]


def test_unknown_category_and_non_numeric_are_rejected():
    bad = _sample()
    bad.loc[0, "smoking_status"] = "sometimes"
    bad.loc[1, "age"] = "forty"
    records, errors, _ = validate_records(bad)
    assert records is None
    assert any("smoking_status" in e for e in errors)
    assert any("`age`" in e for e in errors)


def test_missing_bmi_is_a_warning_not_an_error():
    records, errors, warnings = validate_records(_sample())
    assert not errors and any("BMI missing" in w for w in warnings)


@needs_model
def test_scored_export_round_trips(model):
    records, _, _ = validate_records(_sample())
    risk = model.predict_proba(records)
    assert ((risk >= 0) & (risk <= 1)).all()
    out = pd.DataFrame({"id": records["id"], "risk_score": risk})
    back = pd.read_csv(pd.io.common.StringIO(out.to_csv(index=False)))
    assert np.allclose(back.risk_score, out.risk_score)


@needs_model
def test_dashboard_renders_all_tabs_with_notice():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    assert any("not a medical device" in w.value for w in at.warning)
    assert len(at.tabs) == 4
    assert at.metric[0].value == "6"  # built-in sample scored
