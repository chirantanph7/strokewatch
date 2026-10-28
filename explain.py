"""
StrokeWatch -- SHAP explanations for the adopted model (U5.1-U5.3).

    python explain.py     # global importance figure + reports/shap_global.csv

SHAP values are computed on the model's raw log-odds output, before Platt
calibration. Calibration is a monotone map, so it never changes which
features push a score up or down, only the scale of the final probability.

Two layers of names:
  - transformed columns ("smoking_status_smokes", "age_band_senior") are what
    the model sees and what feature_names.json lists;
  - source variables ("Smoking status", "Age") are what a reader understands.
    One-hot and derived columns are summed back to their source variable, which
    is valid because SHAP values are additive.

SHAP shows how the MODEL uses a feature. It is not a causal or clinical claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from features import BINARY_COLUMNS, NOMINAL_COLUMNS, NUMERIC_COLUMNS

READABLE = {
    "age": "Age",
    "avg_glucose_level": "Average glucose",
    "bmi": "BMI",
    "bmi_missing": "BMI not recorded",
    "hypertension": "Hypertension",
    "heart_disease": "Heart disease",
    "gender": "Gender",
    "work_type": "Work type",
    "Residence_type": "Residence",
    "smoking_status": "Smoking status",
    "ever_married": "Ever married",
    "comorbidity_count": "Comorbidity count",
    "age_x_hypertension": "Age x hypertension",
}

# derived columns are explained under the variable they were built from
DERIVED_SOURCE = {"age_band": "age", "bmi_category": "bmi", "glucose_band": "avg_glucose_level"}

DISCLAIMER = (
    "Contributions show how this model uses each input to produce its score. "
    "They are not causes, and they are not a clinical assessment."
)


def source_variable(column: str) -> str:
    """Transformed column -> source variable it came from."""
    if column in NUMERIC_COLUMNS + BINARY_COLUMNS + ["comorbidity_count", "age_x_hypertension"]:
        return column
    for prefix, src in DERIVED_SOURCE.items():
        if column.startswith(prefix + "_"):
            return src
    for nominal in NOMINAL_COLUMNS:
        if column.startswith(nominal + "_"):
            return nominal
    raise KeyError(f"no source variable for feature column {column!r}")


def readable(variable: str) -> str:
    return READABLE.get(variable, variable)


class Explainer:
    """Wraps the right SHAP explainer for the adopted model's estimator."""

    def __init__(self, stroke_model, background_X: pd.DataFrame, max_background: int = 200, seed: int = 42):
        self.model = stroke_model
        self.estimator = stroke_model.pipeline.named_steps["model"]
        bg = stroke_model.transform_features(background_X)
        if len(bg) > max_background:
            bg = bg.sample(max_background, random_state=seed)
        self.background = bg
        if isinstance(self.estimator, LogisticRegression):
            self._explainer = shap.LinearExplainer(self.estimator, shap.maskers.Independent(bg, max_samples=len(bg)))
        else:
            self._explainer = shap.TreeExplainer(
                self.estimator, data=bg, model_output="raw", feature_perturbation="interventional"
            )

    @property
    def base_value(self) -> float:
        ev = self._explainer.expected_value
        return float(np.ravel(ev)[-1])

    def margin(self, X_features: pd.DataFrame) -> np.ndarray:
        """Raw model output in log-odds, the quantity SHAP decomposes."""
        if isinstance(self.estimator, LogisticRegression):
            return self.estimator.decision_function(X_features)
        return self.estimator.predict(X_features, output_margin=True)

    def shap_columns(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """SHAP value per transformed column, one row per record."""
        Xf = self.model.transform_features(X_raw)
        values = self._explainer.shap_values(Xf)
        if isinstance(values, list):  # some explainers return one array per class
            values = values[-1]
        return pd.DataFrame(np.asarray(values), columns=Xf.columns, index=X_raw.index)

    def shap_sources(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """SHAP grouped back to source variables (sums are exact by additivity)."""
        cols = self.shap_columns(X_raw)
        return cols.T.groupby(lambda c: source_variable(c)).sum().T

    def top_contributions(self, X_raw: pd.DataFrame, k: int = 5) -> list[list[dict]]:
        """Per record: the k source variables that moved the score most."""
        grouped = self.shap_sources(X_raw)
        out = []
        for idx, row in grouped.iterrows():
            top = row.reindex(row.abs().sort_values(ascending=False).index)[:k]
            out.append([
                {
                    "variable": var,
                    "label": readable(var),
                    "value": _display_value(X_raw.loc[idx], var),
                    "shap": float(v),
                    "direction": "raises" if v > 0 else "lowers",
                }
                for var, v in top.items()
            ])
        return out

    def global_importance(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """Mean |SHAP| per source variable over a set of records (U5.2)."""
        grouped = self.shap_sources(X_raw)
        imp = grouped.abs().mean().sort_values(ascending=False)
        return pd.DataFrame({"variable": imp.index, "label": [readable(v) for v in imp.index],
                             "mean_abs_shap": imp.values})


def _display_value(record: pd.Series, var: str):
    if var == "comorbidity_count":
        return int(record["hypertension"]) + int(record["heart_disease"])
    if var == "age_x_hypertension":
        return f"age {record['age']:.0f}, hypertension {int(record['hypertension'])}"
    if var == "bmi_missing":
        return "yes" if pd.isna(record.get("bmi")) else "no"
    val = record.get(var)
    if var in ("hypertension", "heart_disease"):
        return "yes" if int(val) == 1 else "no"
    if isinstance(val, (float, np.floating)):
        return "not recorded" if np.isnan(val) else round(float(val), 1)
    return str(val)


def main():
    import joblib

    import report
    from dataset import ROOT, dev_frame, load_splits, load_typed

    model = joblib.load(ROOT / "models" / "final_model.joblib")
    df, splits = load_typed(), load_splits()
    X_dev, _ = dev_frame(df, splits)
    explainer = Explainer(model, X_dev)
    # the dashboard explains new records against this same background sample
    X_dev.sample(200, random_state=42).to_parquet(ROOT / "models" / "shap_background.parquet", index=False)

    imp = explainer.global_importance(X_dev)
    imp.to_csv(ROOT / "reports" / "shap_global.csv", index=False)

    import matplotlib.pyplot as plt

    report._style()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.barh(imp.label, imp.mean_abs_shap, color=report.COLORS.get(model.name, report.COLORS["logreg"]), height=0.6)
    ax.invert_yaxis()
    for i, v in enumerate(imp.mean_abs_shap):
        ax.text(v, i, f" {v:.2f}", va="center", color=report.INK, fontsize=9)
    ax.set(xlabel="Mean |SHAP| (log-odds), dev set", title="What the model relies on")
    ax.grid(axis="y", visible=False)
    report._save(fig, "shap_global.png")
    print(imp.to_string(index=False))


if __name__ == "__main__":
    main()
