"""
StrokeWatch -- end-to-end experiment run.

    python train.py            # full run (~minutes on a laptop)
    python train.py --quick    # small search budgets, for smoke tests

Stages, in the order the protocol fixes them (docs/evaluation_protocol.md):
  1. Reference points: chance PR-AUC per fold (U3.3)
  2. Feature ablation, raw vs engineered, logistic regression (U2.7)
  3. Baseline, XGBoost and random forest on the 5 frozen folds (U3.2, U4.1-4.3)
  4. Paired benchmark and the pre-registered adoption rule (U4.4, U4.6)
  5. Calibration and threshold on dev out-of-fold predictions (U4.5, U3.5)
  6. Final refit on the whole dev set, then ONE holdout evaluation
Outputs: models/final_model.joblib, models/cohort_oof.parquet,
reports/metrics.json, reports/results.md, reports/search_log.csv, reports/figures/.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

import evaluate as ev
import models as m
from dataset import ROOT, assert_no_holdout, dev_frame, fold_frames, holdout_frame, load_splits, load_typed

TARGET_RECALL = 0.80  # provisional team target (proposal slide 13); see threshold policy
ABLATION_CONFIGS = {
    "raw": {"bands": False, "glucose_bands": False, "comorbidity": False},
    "+bands": {"bands": True, "glucose_bands": False, "comorbidity": False},
    "+glucose_bands": {"bands": False, "glucose_bands": True, "comorbidity": False},
    "+comorbidity": {"bands": False, "glucose_bands": False, "comorbidity": True},
    "all_engineered": {"bands": True, "glucose_bands": True, "comorbidity": True},
}
GROUP_OF = {"+bands": "bands", "+glucose_bands": "glucose_bands", "+comorbidity": "comorbidity"}
MODEL_LABELS = {"logreg": "Logistic regression (baseline)", "xgboost": "XGBoost", "random_forest": "Random forest"}

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def n_folds(splits) -> int:
    return len(splits["folds"])


# ---------------------------------------------------------------- stage 1
def chance_reference(df, splits) -> dict:
    """A random scorer's expected PR-AUC is the prevalence of the fold."""
    prev = [float(fold_frames(df, splits, k)[3].mean()) for k in range(n_folds(splits))]
    return {"per_fold_prevalence": prev, "mean_chance_pr_auc": float(np.mean(prev))}


# ---------------------------------------------------------------- stage 2
def run_ablation(df, splits, quick: bool) -> dict:
    """U2.7: every config scored by the same tuned logistic regression on the
    same folds. A fixed (untuned) XGBoost is shown for context only."""
    rows = []
    for name, cfg in ABLATION_CONFIGS.items():
        for k in range(n_folds(splits)):
            X_tr, y_tr, X_va, y_va = fold_frames(df, splits, k)
            assert_no_holdout(X_tr["id"], splits)
            lr = m.logistic_search(cfg).fit(X_tr, y_tr)
            xgb = m.make_pipeline(
                m.xgb_estimator(y_tr, n_estimators=150 if quick else 300, max_depth=3,
                                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
                cfg,
            ).fit(X_tr, y_tr)
            rows.append({
                "config": name, "fold": k,
                "n_features": len(lr.best_estimator_.named_steps["features"][-1].get_feature_names_out()),
                "logreg_pr_auc": ev.pr_auc(y_va, lr.predict_proba(X_va)[:, 1]),
                "xgb_fixed_pr_auc": ev.pr_auc(y_va, xgb.predict_proba(X_va)[:, 1]),
            })
        log(f"ablation {name}: mean LR PR-AUC "
            f"{np.mean([r['logreg_pr_auc'] for r in rows if r['config'] == name]):.4f}")
    table = pd.DataFrame(rows)

    # Pre-registered selection rule (protocol section 4): a group is kept only
    # if adding it to raw raises mean LR PR-AUC AND wins on >= 3 of 5 folds.
    raw = table[table.config == "raw"].set_index("fold")["logreg_pr_auc"]
    decisions = {}
    for name, group in GROUP_OF.items():
        cand = table[table.config == name].set_index("fold")["logreg_pr_auc"]
        delta = cand - raw
        decisions[group] = {
            "mean_delta": float(delta.mean()),
            "folds_won": int((delta > 0).sum()),
            "keep": bool(delta.mean() > 0 and (delta > 0).sum() >= 3),
        }
    final_cfg = {g: d["keep"] for g, d in decisions.items()}
    log(f"ablation decision: {final_cfg}")

    summary = (
        table.groupby("config", sort=False)
        .agg(n_features=("n_features", "first"),
             logreg_mean=("logreg_pr_auc", "mean"), logreg_sd=("logreg_pr_auc", "std"),
             xgb_fixed_mean=("xgb_fixed_pr_auc", "mean"))
        .reset_index()
    )
    return {"per_fold": table, "summary": summary, "decisions": decisions, "final_config": final_cfg}


# ---------------------------------------------------------------- stage 3
def make_search(name, y_tr, cfg, quick):
    if name == "logreg":
        return m.logistic_search(cfg)
    if name == "xgboost":
        return m.xgb_search(y_tr, cfg, n_iter=8 if quick else m.XGB_BUDGET)
    return m.rf_search(cfg, n_iter=4 if quick else m.RF_BUDGET)


def run_cv(df, splits, cfg, quick: bool):
    """Out-of-fold predictions for each model on the 5 frozen folds."""
    oof = []
    search_log = []
    fold_metrics = []
    for k in range(n_folds(splits)):
        X_tr, y_tr, X_va, y_va = fold_frames(df, splits, k)
        assert_no_holdout(X_tr["id"], splits)
        assert_no_holdout(X_va["id"], splits)
        fold_pred = pd.DataFrame({"id": X_va["id"].values, "fold": k, "y": y_va.values})
        for name in MODEL_LABELS:
            t0 = time.time()
            search = make_search(name, y_tr, cfg, quick).fit(X_tr, y_tr)
            p = search.predict_proba(X_va)[:, 1]
            fold_pred[name] = p
            fold_metrics.append({
                "model": name, "fold": k,
                "pr_auc": ev.pr_auc(y_va, p), "roc_auc": ev.roc_auc(y_va, p),
                "best_params": {kk.replace("model__", ""): (float(v) if isinstance(v, (np.floating, float)) else v)
                                for kk, v in search.best_params_.items()},
            })
            res = pd.DataFrame(search.cv_results_)
            for _, r in res.iterrows():
                search_log.append({
                    "model": name, "outer_fold": k,
                    "params": json.dumps({kk.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
                                          for kk, v in r["params"].items()}),
                    "inner_mean_pr_auc": r["mean_test_score"], "inner_std": r["std_test_score"],
                })
            log(f"fold {k} {name}: val PR-AUC {fold_metrics[-1]['pr_auc']:.4f} ({time.time() - t0:.0f}s)")
        oof.append(fold_pred)
    return pd.concat(oof, ignore_index=True), pd.DataFrame(fold_metrics), pd.DataFrame(search_log)


# ---------------------------------------------------------------- stage 4
def benchmark(oof: pd.DataFrame, fold_metrics: pd.DataFrame) -> dict:
    """U4.4 + U4.6. Adoption rule, quoted verbatim from proposal slide 19:
    'Adopt only if PR-AUC gain CI excludes 0 and calibration not worse; else keep baseline.'"""
    y = oof["y"].values
    calibrated = {name: m.crossfit_calibrate(oof[name].values, y) for name in MODEL_LABELS}
    pooled = {
        name: {
            "pr_auc": ev.bootstrap_ci(y, oof[name].values, ev.pr_auc),
            "roc_auc": ev.bootstrap_ci(y, oof[name].values, ev.roc_auc),
            "brier_calibrated": ev.bootstrap_ci(y, calibrated[name], ev.brier),
            "brier_raw": ev.brier(y, oof[name].values),
        }
        for name in MODEL_LABELS
    }
    per_fold = fold_metrics.pivot(index="fold", columns="model", values="pr_auc")

    candidates = {}
    for cand in ["xgboost", "random_forest"]:
        delta = ev.paired_bootstrap_delta(y, oof["logreg"].values, oof[cand].values)
        brier_delta = ev.paired_bootstrap_delta(y, calibrated["logreg"], calibrated[cand], metric=ev.brier)
        gain_excludes_zero = delta["ci_low"] > 0
        calib_not_worse = brier_delta["delta"] <= 0
        candidates[cand] = {
            "pr_auc_delta_vs_baseline": delta,
            "brier_delta_vs_baseline": brier_delta,
            "fold_deltas": (per_fold[cand] - per_fold["logreg"]).round(4).tolist(),
            "folds_won": int((per_fold[cand] > per_fold["logreg"]).sum()),
            "gain_ci_excludes_zero": bool(gain_excludes_zero),
            "calibration_not_worse": bool(calib_not_worse),
            "passes_rule": bool(gain_excludes_zero and calib_not_worse),
        }
    passing = [c for c, r in candidates.items() if r["passes_rule"]]
    adopted = max(passing, key=lambda c: candidates[c]["pr_auc_delta_vs_baseline"]["delta"]) if passing else "logreg"
    log(f"adoption decision: {adopted}")
    return {"pooled": pooled, "per_fold": per_fold, "candidates": candidates,
            "adopted": adopted, "calibrated_oof": calibrated}


# ---------------------------------------------------------------- stage 6
def refit_on_dev(name, df, splits, cfg, quick):
    X_dev, y_dev = dev_frame(df, splits)
    assert_no_holdout(X_dev["id"], splits)
    search = make_search(name, y_dev, cfg, quick).fit(X_dev, y_dev)
    return search.best_estimator_, search.best_params_


def main(quick: bool = False) -> dict:
    t_start = time.time()
    df = load_typed()
    splits = load_splits()
    (REPORTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    log("stage 1: chance reference")
    chance = chance_reference(df, splits)

    log("stage 2: feature ablation (U2.7)")
    ablation = run_ablation(df, splits, quick)
    cfg = ablation["final_config"]

    log(f"stage 3: cross-validated models on feature config {cfg}")
    oof, fold_metrics, search_log = run_cv(df, splits, cfg, quick)

    log("stage 4: paired benchmark and adoption rule")
    bench = benchmark(oof, fold_metrics)
    adopted = bench["adopted"]

    log("stage 5: calibration and threshold on dev out-of-fold predictions")
    y_oof = oof["y"].values
    calibrator = m.PlattCalibrator().fit(oof[adopted].values, y_oof)
    p_oof = calibrator.transform(oof[adopted].values)
    threshold = ev.threshold_for_recall(y_oof, p_oof, TARGET_RECALL)
    dev_summary = ev.summarize(y_oof, p_oof, threshold)
    sweep = ev.threshold_sweep(y_oof, p_oof)

    log(f"stage 6: refit {adopted} on full dev set")
    pipeline, best_params = refit_on_dev(adopted, df, splits, cfg, quick)
    final = m.StrokeModel(adopted, pipeline, calibrator, threshold, TARGET_RECALL, cfg)

    log("stage 6: ONE evaluation on the locked holdout")
    X_ho, y_ho = holdout_frame(df, splits, unlock=True)
    p_ho = final.predict_proba(X_ho)
    holdout = ev.summarize(y_ho.values, p_ho, threshold)
    holdout["calibration"] = ev.calibration_table(y_ho.values, p_ho, n_bins=5).to_dict("list")

    joblib.dump(final, MODELS_DIR / "final_model.joblib")
    cohort = oof[["id", "fold", "y"]].assign(score=p_oof)
    cohort.to_parquet(MODELS_DIR / "cohort_oof.parquet", index=False)
    pd.DataFrame({"id": X_ho["id"].values, "y": y_ho.values, "score": p_ho}).to_parquet(
        MODELS_DIR / "holdout_scores.parquet", index=False)
    oof.to_csv(REPORTS_DIR / "oof_predictions.csv", index=False)
    search_log.to_csv(REPORTS_DIR / "search_log.csv", index=False)

    results = {
        "run": {
            "quick": quick,
            "seconds": round(time.time() - t_start),
            "python": platform.python_version(),
            "sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "pandas": pd.__version__,
            "n_dev": len(splits["dev_ids"]), "n_holdout": len(splits["holdout_ids"]),
            "target_recall": TARGET_RECALL,
        },
        "chance": chance,
        "ablation": {
            "summary": ablation["summary"].to_dict("records"),
            "per_fold": ablation["per_fold"].to_dict("records"),
            "decisions": ablation["decisions"],
            "final_config": cfg,
        },
        "cv": {
            "fold_metrics": fold_metrics.to_dict("records"),
            "pooled": bench["pooled"],
            "candidates": bench["candidates"],
        },
        "adopted": adopted,
        "final_best_params": {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
                              for k, v in best_params.items()},
        "feature_names": final.feature_names,
        "threshold": threshold,
        "dev_oof": dev_summary,
        "threshold_sweep": sweep.to_dict("records"),
        "holdout": holdout,
    }
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    import report  # writes results.md and figures from metrics + OOF predictions
    report.write_all(results, oof, bench["calibrated_oof"], cohort)
    log(f"done in {time.time() - t_start:.0f}s. Adopted {adopted}; holdout PR-AUC "
        f"{holdout['pr_auc']['value']:.3f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small search budgets (smoke test)")
    main(quick=parser.parse_args().quick)
