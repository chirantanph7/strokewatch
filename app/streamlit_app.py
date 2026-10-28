"""
StrokeWatch research-prototype dashboard (U5.3-U5.6).

    streamlit run app/streamlit_app.py

Four tabs, with the non-clinical notice above all of them:
  Score records     CSV upload -> schema check -> calibrated risk + flag -> scored CSV
  Record inspector  one record's inputs, score, flag and top-5 SHAP contributions
  Cohort view       dev out-of-fold score distribution and precision-recall curve
  Model card        intended use, performance, limitations
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from explain import DISCLAIMER, Explainer  # noqa: E402
from validation import validate_records  # noqa: E402

MODEL_PATH = ROOT / "models" / "final_model.joblib"
NOTICE = (
    "**Research prototype, not a medical device.** Built for a university course on a public, "
    "de-identified dataset with no external validation. Do not use it to diagnose, triage or "
    "make any decision about a real person."
)
LABELS = {"logreg": "Logistic regression", "xgboost": "XGBoost", "random_forest": "Random forest"}

st.set_page_config(page_title="StrokeWatch", layout="wide")


@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists():
        return None
    model = joblib.load(MODEL_PATH)
    background = pd.read_parquet(ROOT / "models" / "shap_background.parquet")
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    cohort = pd.read_parquet(ROOT / "models" / "cohort_oof.parquet")
    return model, Explainer(model, background), metrics, cohort


def score(model, explainer, records: pd.DataFrame) -> pd.DataFrame:
    risk = model.predict_proba(records)
    top = explainer.top_contributions(records, k=3)
    out = records[["id"]].copy()
    out["risk_score"] = np.round(risk, 4)
    out["flagged"] = (risk >= model.threshold).astype(int)
    for i in range(3):
        out[f"factor_{i + 1}"] = [
            f"{t[i]['label']} = {t[i]['value']} ({t[i]['direction']} score)" if len(t) > i else "" for t in top
        ]
    return out


def tab_score(model, explainer):
    st.subheader("Score records")
    st.caption("Upload a CSV with the 10 predictor columns of the Kaggle stroke dataset "
               "(`id` optional, `stroke` ignored). Values are checked against the project schema first.")
    source = st.radio("Records", ["Use the built-in synthetic sample", "Upload a CSV"], horizontal=True)
    if source == "Upload a CSV":
        upload = st.file_uploader("CSV file", type=["csv"])
        if upload is None:
            return
        raw = pd.read_csv(upload, dtype=str, keep_default_na=False)
    else:
        raw = pd.read_csv(ROOT / "app" / "sample_records.csv", dtype=str, keep_default_na=False)

    records, errors, warnings = validate_records(raw)
    for e in errors:
        st.error(e)
    for w in warnings:
        st.warning(w)
    if records is None:
        st.info("Fix the problems above and upload again. Nothing was scored.")
        return

    scored = score(model, explainer, records)
    st.session_state["records"] = records
    st.session_state["scored"] = scored
    n_flag = int(scored.flagged.sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Records scored", len(scored))
    c2.metric("Flagged for review", n_flag)
    c3.metric("Flag threshold (calibrated risk)", f"{model.threshold:.3f}")
    st.dataframe(scored, width="stretch", hide_index=True)
    st.download_button("Download scored CSV", scored.to_csv(index=False).encode(), "scored_output.csv", "text/csv")
    st.caption(f"Flag = calibrated risk at or above the threshold set on development data to catch "
               f"{model.target_recall:.0%} of strokes. Most flagged records will not be strokes. {DISCLAIMER}")


def tab_inspect(model, explainer):
    st.subheader("Record inspector")
    if "records" not in st.session_state:
        st.info("Score some records in the first tab, then pick one here.")
        return
    records, scored = st.session_state["records"], st.session_state["scored"]
    rid = st.selectbox("Record id", scored["id"].tolist())
    rec = records[records["id"] == rid]
    row = scored[scored["id"] == rid].iloc[0]

    left, right = st.columns([1, 2])
    with left:
        st.metric("Calibrated risk", f"{row.risk_score:.1%}")
        st.markdown("**Flagged for review**" if row.flagged else "Not flagged")
        shown = rec.drop(columns=["_flags"]).T.rename(columns=lambda _: "value").astype(str)
        st.dataframe(shown, width="stretch")
    with right:
        top = explainer.top_contributions(rec, k=5)[0]
        st.markdown("**Top 5 contributions to this score**")
        chart = pd.DataFrame({
            "factor": [f"{t['label']} = {t['value']}" for t in top],
            "contribution": [t["shap"] for t in top],
            "direction": [t["direction"] for t in top],
        })
        bars = alt.Chart(chart).mark_bar(cornerRadiusEnd=4, height=18).encode(
            x=alt.X("contribution:Q", title="contribution to score (log-odds)"),
            y=alt.Y("factor:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["raises", "lowers"], range=["#eb6834", "#2a78d6"]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["factor", alt.Tooltip("contribution:Q", format="+.2f")],
        )
        st.altair_chart(bars + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#52514e").encode(x="x:Q"),
                        width="stretch")
        for t in top:
            st.markdown(f"- **{t['label']}** = {t['value']} {t['direction']} the score by {abs(t['shap']):.2f} log-odds")
        st.caption(DISCLAIMER)


def tab_cohort(model, metrics, cohort):
    st.subheader("Cohort view: development set, out-of-fold")
    st.caption("Each dev patient scored by a model that never saw them during training.")
    from sklearn.metrics import precision_recall_curve

    c1, c2 = st.columns(2)
    with c1:
        # share of each group per bin, so the 5% stroke group is visible next to the 95%
        bins = np.linspace(0, cohort.score.quantile(0.99), 21)
        rows = []
        for label, part in [("no stroke", cohort.score[cohort.y == 0]), ("stroke", cohort.score[cohort.y == 1])]:
            counts = np.histogram(part.clip(upper=bins[-1]), bins)[0]
            rows += [{"risk": round(float(b), 3), "group": label, "share": c / len(part)} for b, c in zip(bins[:-1], counts)]
        st.markdown("**Score distribution (share of each group)**")
        hist = alt.Chart(pd.DataFrame(rows)).mark_bar(opacity=0.75, binSpacing=1).encode(
            x=alt.X("risk:Q", bin=alt.Bin(binned=True, step=float(bins[1] - bins[0])), title="calibrated risk"),
            y=alt.Y("share:Q", stack=None, axis=alt.Axis(format="%"), title="share of group"),
            color=alt.Color("group:N", scale=alt.Scale(domain=["no stroke", "stroke"], range=["#2a78d6", "#eb6834"]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["group", "risk", alt.Tooltip("share:Q", format=".1%")],
        )
        rule = alt.Chart(pd.DataFrame({"t": [model.threshold]})).mark_rule(color="#0b0b0b", strokeDash=[4, 3]).encode(x="t:Q")
        st.altair_chart(hist + rule, width="stretch")
        st.caption("Dashed line: flag threshold.")
    with c2:
        prec, rec, _ = precision_recall_curve(cohort.y, cohort.score)
        st.markdown("**Precision-recall curve**")
        st.line_chart(pd.DataFrame({"recall": rec, "precision": prec}), x="recall", y="precision")
    at = metrics["dev_oof"]["at_threshold"]
    st.markdown(f"At the threshold: recall **{at['recall']:.2f}**, precision **{at['precision']:.2f}**, "
                f"{at['flag_rate']:.0%} of patients flagged. Chance PR-AUC is {metrics['dev_oof']['prevalence']:.3f}.")


def tab_card(model, metrics):
    st.subheader("Model card and limitations")
    ho = metrics["holdout"]
    st.markdown((ROOT / "docs" / "model_card.md").read_text())
    st.markdown(f"**Locked holdout ({ho['n']} patients, {ho['positives']} strokes):** "
                f"PR-AUC {ho['pr_auc']['value']:.3f} (95% CI {ho['pr_auc']['ci_low']:.3f}-{ho['pr_auc']['ci_high']:.3f}), "
                f"ROC-AUC {ho['roc_auc']['value']:.3f}, recall at threshold {ho['at_threshold']['recall']:.2f}, "
                f"precision {ho['at_threshold']['precision']:.2f}.")
    st.markdown((ROOT / "docs" / "limitations.md").read_text())


def main():
    st.title("StrokeWatch")
    st.warning(NOTICE)
    assets = load_assets()
    if assets is None:
        st.error("No trained model found. Run `make all` (or `python train.py && python explain.py`) first.")
        return
    model, explainer, metrics, cohort = assets
    st.caption(f"Model: {LABELS.get(model.name, model.name)} with Platt calibration, "
               f"{len(model.feature_names)} engineered features.")
    t1, t2, t3, t4 = st.tabs(["Score records", "Record inspector", "Cohort view", "Model card"])
    with t1:
        tab_score(model, explainer)
    with t2:
        tab_inspect(model, explainer)
    with t3:
        tab_cohort(model, metrics, cohort)
    with t4:
        tab_card(model, metrics)


main()
