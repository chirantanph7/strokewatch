"""
StrokeWatch -- shared evaluation library (M3 protocol).

Every model is scored with these functions and nothing else, so comparisons
are like for like. Accuracy is deliberately absent: at ~4.9% prevalence,
predicting "no stroke" for everyone scores ~95%.

  U3.3  Chance reference: PR-AUC of a random scorer equals prevalence.
  U3.4  PR-AUC (primary), ROC-AUC, Brier, and recall / precision /
        specificity at a threshold, with bootstrap 95% CIs.
  U3.5  Threshold policy: the highest threshold that still meets the target
        recall on validation predictions. Chosen on dev data only.
  U4.4  Patient-level paired bootstrap for model-vs-model PR-AUC deltas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

N_BOOT = 1000
SEED = 42


def pr_auc(y, p) -> float:
    return float(average_precision_score(y, p))


def roc_auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def brier(y, p) -> float:
    return float(brier_score_loss(y, p))


def threshold_metrics(y, p, threshold: float) -> dict:
    """Confusion-matrix metrics when scores >= threshold are flagged."""
    y = np.asarray(y)
    flag = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, flag, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "threshold": float(threshold),
        "recall": float(recall),
        "precision": float(precision),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "flag_rate": float(flag.mean()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def threshold_for_recall(y, p, target_recall: float) -> float:
    """Highest threshold whose recall is still >= target (U3.5).
    Highest, because among thresholds that meet the recall target it flags
    the fewest people and so has the best precision."""
    y = np.asarray(y)
    p = np.asarray(p)
    pos_scores = np.sort(p[y == 1])[::-1]
    if len(pos_scores) == 0:
        raise ValueError("no positives to set a recall threshold on")
    k = int(np.ceil(target_recall * len(pos_scores)))  # positives that must be caught
    return float(pos_scores[max(k, 1) - 1])


def threshold_sweep(y, p, recall_targets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95)) -> pd.DataFrame:
    """Trade-off table: what each recall target costs in precision and flags."""
    rows = []
    for r in recall_targets:
        t = threshold_for_recall(y, p, r)
        rows.append({"target_recall": r, **threshold_metrics(y, p, t)})
    return pd.DataFrame(rows)


def _boot_indices(y, n_boot: int, seed: int):
    """Bootstrap resamples, stratified so every resample keeps some positives."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def bootstrap_ci(y, p, metric, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """(point estimate, lower 2.5%, upper 97.5%) for metric(y, p)."""
    y, p = np.asarray(y), np.asarray(p)
    stats = [metric(y[i], p[i]) for i in _boot_indices(y, n_boot, seed)]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(metric(y, p)), float(lo), float(hi)


def paired_bootstrap_delta(y, p_a, p_b, metric=pr_auc, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """metric(B) - metric(A) on the same resampled patients each time, so the
    CI reflects the difference between models, not each model's own noise."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    deltas = np.array([metric(y[i], p_b[i]) - metric(y[i], p_a[i]) for i in _boot_indices(y, n_boot, seed)])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta": float(metric(y, p_b) - metric(y, p_a)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_b_better": float((deltas > 0).mean()),
    }


def summarize(y, p, threshold: float | None = None, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Headline metrics with CIs, plus threshold metrics if a threshold is given."""
    out = {
        "n": int(len(y)),
        "positives": int(np.sum(y)),
        "prevalence": float(np.mean(y)),
    }
    for name, fn in [("pr_auc", pr_auc), ("roc_auc", roc_auc), ("brier", brier)]:
        est, lo, hi = bootstrap_ci(y, p, fn, n_boot, seed)
        out[name] = {"value": est, "ci_low": lo, "ci_high": hi}
    if threshold is not None:
        out["at_threshold"] = threshold_metrics(y, p, threshold)
        for name in ["recall", "precision"]:
            fn = lambda yy, pp, k=name: threshold_metrics(yy, pp, threshold)[k]  # noqa: E731
            est, lo, hi = bootstrap_ci(y, p, fn, n_boot, seed)
            out["at_threshold"][f"{name}_ci"] = [lo, hi]
    return out


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Reliability curve points with quantile bins (equal-width bins would put
    almost every patient in the first bin at this prevalence)."""
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "observed_rate": frac_pos})
