"""
U2.6 -- Generates feature_dictionary.md and feature_names.json from the fitted
FeatureTransformer (run on Jyot's real data), so the docs can't drift from the code.

If train.py has produced models/final_model.joblib, the docs describe exactly
that model's features (fit on the dev set). Otherwise the default transformer
is fit on the dev set. Either way, nothing is fit on the holdout.

Run with: python generate_docs.py
"""

import json

import joblib

from dataset import ROOT, dev_frame, load_splits, load_typed
from features import build_feature_transformer
from upstream_preprocess import CleanTransformer

SOURCE_MAP = {
    "gender": ("gender", "one-hot encoded"),
    "work_type": ("work_type", "one-hot encoded"),
    "Residence_type": ("Residence_type", "one-hot encoded"),
    "smoking_status": ("smoking_status", "one-hot encoded"),
    "ever_married": ("ever_married", "one-hot encoded"),
    "age_band": ("age", "derived: age band (child <18, adult 18-39, middle_age 40-59, senior 60+), one-hot encoded"),
    "bmi_category": ("bmi", "derived: WHO adult BMI category (underweight/normal/overweight/obese), one-hot encoded"),
    "age": ("age", "standard-scaled (fit on training fold only)"),
    "avg_glucose_level": ("avg_glucose_level", "standard-scaled (fit on training fold only)"),
    "bmi": ("bmi", "M1-imputed (train-fold median) then standard-scaled (fit on training fold only)"),
    "hypertension": ("hypertension", "passthrough binary, no transform"),
    "heart_disease": ("heart_disease", "passthrough binary, no transform"),
    "bmi_missing": ("bmi", "passthrough binary indicator from M1's CleanTransformer"),
    "glucose_band": ("avg_glucose_level", "derived: glucose band (normal <100, elevated 100-139, high 140-199, very_high 200+ mg/dL), one-hot encoded"),
    "comorbidity_count": ("hypertension, heart_disease", "derived: hypertension + heart_disease (0-2), then standard-scaled"),
    "age_x_hypertension": ("age, hypertension", "derived: age in years x hypertension (0 if no hypertension), then standard-scaled"),
}

RATIONALE = {
    "gender": "Demographic risk factor, documented in stroke risk literature.",
    "work_type": "Proxy for activity level and stress exposure.",
    "Residence_type": "Proxy for healthcare access.",
    "smoking_status": "Established modifiable risk factor; Unknown kept as informative missingness, not imputed.",
    "ever_married": "Included for completeness; weak prior evidence of direct causal link.",
    "age_band": "Coarse, readable age grouping for SHAP explanations.",
    "bmi_category": "WHO-standard BMI categories; clinically interpretable, not diagnostic.",
    "age": "Established stroke risk factor; strongest single predictor in literature.",
    "avg_glucose_level": "Established modifiable risk factor (proxy for glycemic control).",
    "bmi": "Established modifiable risk factor.",
    "hypertension": "Established modifiable risk factor.",
    "heart_disease": "Established comorbidity risk factor.",
    "bmi_missing": "Missingness kept as a signal per M1's schema notes, not discarded on impute.",
    "glucose_band": "Heuristic, NON-DIAGNOSTIC bands: the dataset does not say whether glucose is fasting or random.",
    "comorbidity_count": "Combined burden of the two recorded cardiovascular comorbidities.",
    "age_x_hypertension": "Lets a linear model give hypertension more weight at older ages.",
}


def prefix_group(col: str) -> str:
    if col in SOURCE_MAP:
        return col
    # longest prefix wins, so "bmi_missing" isn't shadowed by "bmi"
    candidates = [b for b in SOURCE_MAP if col.startswith(b + "_")]
    return max(candidates, key=len) if candidates else col


def main():
    model_path = ROOT / "models" / "final_model.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
        cols = model.feature_names
        provenance = (f"Describes the adopted model ({model.name}), feature config {model.feature_config}, "
                      "fit on the development set.")
    else:
        X_dev, _ = dev_frame(load_typed(), load_splits())
        X_clean = CleanTransformer().fit_transform(X_dev)
        cols = list(build_feature_transformer().fit(X_clean).transform(X_clean).columns)
        provenance = "No trained model found; describes the default transformer fit on the development set."

    with open(ROOT / "feature_names.json", "w") as f:
        json.dump(cols, f, indent=2)

    lines = [
        "# StrokeWatch feature dictionary",
        "",
        "Auto-generated from `features.py` by `generate_docs.py` -- do not hand-edit.",
        f"Regenerate after any change to the transformer. {len(cols)} output columns.",
        provenance,
        "",
        "| Output column | Source column | Transform | Rationale |",
        "|---|---|---|---|",
    ]
    for col in cols:
        base = prefix_group(col)
        source, transform = SOURCE_MAP.get(base, (base, "unknown"))
        lines.append(f"| `{col}` | `{source}` | {transform} | {RATIONALE.get(base, '')} |")

    lines += [
        "",
        "Which derived groups are in the model is decided by the U2.7 ablation in `train.py`;",
        "see `reports/results.md` section 2.",
    ]
    with open(ROOT / "feature_dictionary.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote feature_names.json and feature_dictionary.md ({len(cols)} columns)")


if __name__ == "__main__":
    main()
