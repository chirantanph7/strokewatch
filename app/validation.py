"""
StrokeWatch dashboard -- schema validation for uploaded CSVs (U5.5).

Checks an upload against M1's data/schema.yaml before anything is scored, and
returns readable messages instead of a stack trace. Hard errors (a missing
column, a value that is not a number, an unknown category) stop scoring;
soft warnings (out-of-range values, missing BMI) are shown but scored, the same
way M1 flags rather than drops.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "data" / "schema.yaml"
REQUIRED_PREDICTORS = [
    "gender", "age", "hypertension", "heart_disease", "ever_married", "work_type",
    "Residence_type", "avg_glucose_level", "bmi", "smoking_status",
]


def load_schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return yaml.safe_load(f)["columns"]


def validate_records(raw: pd.DataFrame, schema: dict | None = None):
    """Returns (typed DataFrame or None, errors, warnings)."""
    schema = schema or load_schema()
    errors, warnings = [], []
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_PREDICTORS if c not in df.columns]
    if missing:
        return None, [f"Missing required column(s): {', '.join(missing)}."], []
    if len(df) == 0:
        return None, ["The file has a header but no records."], []

    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))
        warnings.append("No `id` column, so records were numbered 1..n.")

    for col in REQUIRED_PREDICTORS:
        spec = schema[col]
        if spec["dtype"] == "category" or (spec["dtype"] == "int64" and "categories" in spec):
            allowed = spec["categories"]
            values = df[col]
            if spec["dtype"] == "int64":
                values = pd.to_numeric(values, errors="coerce")
            else:
                values = values.astype(str).str.strip()
            bad = sorted(set(values[~values.isin(allowed)].astype(str)))
            if bad:
                errors.append(f"`{col}` has value(s) {', '.join(bad[:5])}; allowed: {', '.join(map(str, allowed))}.")
            df[col] = values
        else:
            text = df[col].astype(str).str.strip()
            numeric = pd.to_numeric(text.replace({"N/A": None, "": None, "nan": None}), errors="coerce")
            not_number = numeric.isna() & text.notna() & ~text.isin(["N/A", "", "nan", "None"])
            if not_number.any():
                errors.append(f"`{col}` has non-numeric value(s) in row(s) "
                              f"{', '.join(str(i + 1) for i in df.index[not_number][:5])}.")
            if numeric.isna().any() and not spec.get("nullable", False):
                errors.append(f"`{col}` is empty in {int(numeric.isna().sum())} row(s); it is required.")
            if "plausible_range" in spec:
                lo, hi = spec["plausible_range"]
                out = numeric.notna() & ~numeric.between(lo, hi)
                if out.any():
                    warnings.append(f"`{col}` is outside the plausible range {lo}-{hi} in {int(out.sum())} row(s); "
                                    "scored, but treat those scores with extra caution.")
            df[col] = numeric

    if errors:
        return None, errors, warnings
    if df["bmi"].isna().any():
        warnings.append(f"BMI missing in {int(df['bmi'].isna().sum())} row(s); imputed with the training median "
                        "and flagged through `bmi_missing`, as in training.")

    for col in ["hypertension", "heart_disease"]:
        df[col] = df[col].astype("int64")
    for col in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]:
        df[col] = df[col].astype("category")
    df["_flags"] = ""
    return df[["id"] + REQUIRED_PREDICTORS + ["_flags"]], errors, warnings
