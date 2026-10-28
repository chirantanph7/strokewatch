"""
StrokeWatch -- writes reports/results.md and reports/figures/*.png from a
train.py run. Called by train.py; never computes a new metric itself, it only
formats what train.py measured.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

import evaluate as ev  # noqa: E402
from dataset import ROOT  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures"

# Reference data-viz palette (light mode): fixed slot per model, never by rank
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
COLORS = {"logreg": "#2a78d6", "xgboost": "#eb6834", "random_forest": "#1baf7a"}
LABELS = {"logreg": "Logistic regression", "xgboost": "XGBoost", "random_forest": "Random forest"}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "legend.frameon": False, "legend.labelcolor": INK,
        "lines.linewidth": 2,
    })


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=200)
    plt.close(fig)


def fig_pr_curves(oof, prevalence):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for name in LABELS:
        prec, rec, _ = precision_recall_curve(oof["y"], oof[name])
        ax.plot(rec, prec, color=COLORS[name], label=f"{LABELS[name]} (PR-AUC {ev.pr_auc(oof['y'], oof[name]):.3f})")
    ax.axhline(prevalence, color=INK_2, ls="--", lw=1.2)
    ax.text(0.01, prevalence + 0.01, f"chance = prevalence ({prevalence:.3f})", ha="left", va="bottom",
            color=INK_2, fontsize=9)
    ax.set(xlabel="Recall (share of strokes caught)", ylabel="Precision (share of flags that are strokes)",
           xlim=(0, 1), ylim=(0, 0.6), title="Precision-recall, dev set out-of-fold")
    ax.legend(loc="upper right")
    _save(fig, "pr_curves.png")


def fig_fold_pr_auc(per_fold: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    names = list(LABELS)
    x = np.arange(len(names))
    for fold, row in per_fold.iterrows():
        ax.plot(x, [row[n] for n in names], color=GRID, lw=1.2, zorder=1)
    for i, n in enumerate(names):
        ax.scatter([i] * len(per_fold), per_fold[n], s=60, color=COLORS[n], edgecolor=SURFACE, lw=2, zorder=3)
        ax.scatter([i], [per_fold[n].mean()], marker="_", s=900, color=INK, lw=2, zorder=4)
    ax.set_xticks(x, [LABELS[n] for n in names])
    ax.set(ylabel="Validation PR-AUC", title="PR-AUC on each frozen fold (bar = mean)")
    ax.grid(axis="x", visible=False)
    _save(fig, "fold_pr_auc.png")


def fig_ablation(per_fold: pd.DataFrame, summary: pd.DataFrame):
    """Paired view: each config's PR-AUC minus raw's on the same fold. Showing
    deltas, not absolute bars, makes the fold-to-fold noise visible."""
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    wide = per_fold.pivot(index="fold", columns="config", values="logreg_pr_auc")
    configs = [c for c in summary["config"] if c != "raw"]
    ax.axvline(0, color=INK_2, lw=1.2)
    for i, c in enumerate(configs):
        d = wide[c] - wide["raw"]
        ax.scatter(d, [i] * len(d), s=60, color=COLORS["logreg"], edgecolor=SURFACE, lw=2, zorder=3)
        ax.scatter([d.mean()], [i], marker="|", s=500, color=INK, lw=2.5, zorder=4)
        ax.text(1.02, i, f"mean {d.mean():+.4f}", transform=ax.get_yaxis_transform(), ha="left",
                va="center", color=INK, fontsize=10, clip_on=False)
    ax.set_yticks(range(len(configs)), configs)
    ax.set_ylim(len(configs) - 0.5, -0.5)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
    ax.set(xlabel="PR-AUC change vs raw features, per fold (bar = mean)",
           title="Feature ablation (U2.7), logistic regression")
    ax.grid(axis="y", visible=False)
    _save(fig, "ablation.png")


def fig_calibration(oof, calibrated, adopted):
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    y = oof["y"].values
    raw = ev.calibration_table(y, oof[adopted].values)
    cal = ev.calibration_table(y, calibrated[adopted])
    hi = max(raw.mean_predicted.max(), 0.3)
    ax.plot([0, hi], [0, hi], color=INK_2, ls="--", lw=1.2, label="perfect calibration")
    ax.plot(raw.mean_predicted, raw.observed_rate, color=GRID, marker="o", ms=7, mec=INK_2, label="raw class-weighted score")
    ax.plot(cal.mean_predicted, cal.observed_rate, color=COLORS[adopted], marker="o", ms=8, mec=SURFACE,
            label="after Platt calibration")
    ax.set(xlabel="Mean predicted risk (decile)", ylabel="Observed stroke rate",
           title=f"Calibration, {LABELS[adopted]}")
    ax.legend(loc="upper left")
    _save(fig, "calibration.png")


def fig_threshold(sweep: pd.DataFrame, chosen_recall: float):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(sweep.target_recall, sweep.precision, color=COLORS["logreg"], marker="o", ms=8, mec=SURFACE,
            label="precision")
    ax.plot(sweep.target_recall, sweep.flag_rate, color=COLORS["xgboost"], marker="o", ms=8, mec=SURFACE,
            label="share of patients flagged")
    ax.axvline(chosen_recall, color=INK_2, ls="--", lw=1.2)
    ax.text(chosen_recall, ax.get_ylim()[1] * 0.97, " chosen target", color=INK_2, fontsize=9, va="top")
    ax.set(xlabel="Recall target", ylabel="Proportion", title="What each recall target costs")
    ax.legend(loc="upper left")
    _save(fig, "threshold_tradeoff.png")


def _ci(d):
    return f"{d['value']:.3f} ({d['ci_low']:.3f} to {d['ci_high']:.3f})"


def _ci3(t):
    return f"{t[0]:.3f} ({t[1]:.3f} to {t[2]:.3f})"


def write_markdown(r: dict):
    abl = pd.DataFrame(r["ablation"]["summary"])
    cand = r["cv"]["candidates"]
    pooled = r["cv"]["pooled"]
    ho = r["holdout"]
    dv = r["dev_oof"]
    at = ho["at_threshold"]
    lines = [
        "# StrokeWatch results",
        "",
        "Auto-generated by `train.py` -- do not hand-edit. All numbers come from `reports/metrics.json`.",
        f"Run: Python {r['run']['python']}, scikit-learn {r['run']['sklearn']}, xgboost {r['run']['xgboost']},"
        f" {r['run']['seconds']} s{' (QUICK MODE -- small search budgets)' if r['run']['quick'] else ''}.",
        "",
        "## 1. Reference point",
        f"A random scorer's PR-AUC equals prevalence: **{r['chance']['mean_chance_pr_auc']:.3f}** on the dev folds.",
        "",
        "## 2. Feature ablation (U2.7)",
        "Same tuned logistic regression, same 5 frozen folds. XGBoost with fixed, untuned settings shown for context.",
        "",
        "| Config | Features | LR PR-AUC mean | LR sd | XGB (fixed) PR-AUC |",
        "|---|---|---|---|---|",
    ]
    for _, row in abl.iterrows():
        lines.append(f"| {row.config} | {row.n_features} | {row.logreg_mean:.4f} | {row.logreg_sd:.4f} | {row.xgb_fixed_mean:.4f} |")
    lines += ["", "Selection rule: keep a group only if adding it to raw raises mean LR PR-AUC **and** wins on at least 3 of 5 folds.", "",
              "| Group | Mean delta vs raw | Folds won | Kept |", "|---|---|---|---|"]
    for g, d in r["ablation"]["decisions"].items():
        lines.append(f"| {g} | {d['mean_delta']:+.4f} | {d['folds_won']}/5 | {'yes' if d['keep'] else 'no'} |")
    lines += [
        "",
        f"Final feature set: `{r['ablation']['final_config']}` ({len(r['feature_names'])} columns).",
        "",
        "## 3. Models on the 5 frozen folds (out-of-fold, pooled dev set, bootstrap 95% CI)",
        "",
        "| Model | PR-AUC | ROC-AUC | Brier (calibrated) |",
        "|---|---|---|---|",
    ]
    for name, label in LABELS.items():
        p = pooled[name]
        lines.append(f"| {label} | {_ci3(p['pr_auc'])} | {_ci3(p['roc_auc'])} | {_ci3(p['brier_calibrated'])} |")
    lines += [
        "",
        "## 4. Adoption decision (U4.6)",
        "",
        "> Rule (proposal slide 19, fixed before results): *Adopt only if PR-AUC gain CI excludes 0 and calibration not worse; else keep baseline.*",
        "",
        "| Candidate | PR-AUC gain vs baseline (95% CI) | Folds won | Brier change | Passes |",
        "|---|---|---|---|---|",
    ]
    for c, d in cand.items():
        g = d["pr_auc_delta_vs_baseline"]
        b = d["brier_delta_vs_baseline"]
        lines.append(f"| {LABELS[c]} | {g['delta']:+.4f} ({g['ci_low']:+.4f} to {g['ci_high']:+.4f}) | "
                     f"{d['folds_won']}/5 | {b['delta']:+.5f} | {'yes' if d['passes_rule'] else 'no'} |")
    lines += [
        "",
        f"**Adopted: {LABELS[r['adopted']]}.**",
        "",
        "## 5. Threshold policy (U3.5)",
        f"Target recall {r['run']['target_recall']:.0%} (provisional), chosen on calibrated dev out-of-fold predictions:"
        f" threshold = **{r['threshold']:.4f}** calibrated risk.",
        "",
        "| Target recall | Threshold | Recall | Precision | Specificity | Flagged |",
        "|---|---|---|---|---|---|",
    ]
    for s in r["threshold_sweep"]:
        lines.append(f"| {s['target_recall']:.0%} | {s['threshold']:.4f} | {s['recall']:.3f} | {s['precision']:.3f} | "
                     f"{s['specificity']:.3f} | {s['flag_rate']:.1%} |")
    lines += [
        "",
        "## 6. Locked holdout, evaluated once",
        f"{ho['n']} patients, {ho['positives']} strokes. Model refit on the full dev set; threshold and calibration carried over from dev.",
        "",
        "| Metric | Holdout (95% CI) | Dev out-of-fold |",
        "|---|---|---|",
        f"| PR-AUC | {_ci(ho['pr_auc'])} | {_ci(dv['pr_auc'])} |",
        f"| ROC-AUC | {_ci(ho['roc_auc'])} | {_ci(dv['roc_auc'])} |",
        f"| Brier | {_ci(ho['brier'])} | {_ci(dv['brier'])} |",
        f"| Recall at threshold | {at['recall']:.3f} ({at['recall_ci'][0]:.3f} to {at['recall_ci'][1]:.3f}) | {dv['at_threshold']['recall']:.3f} |",
        f"| Precision at threshold | {at['precision']:.3f} ({at['precision_ci'][0]:.3f} to {at['precision_ci'][1]:.3f}) | {dv['at_threshold']['precision']:.3f} |",
        f"| Specificity | {at['specificity']:.3f} | {dv['at_threshold']['specificity']:.3f} |",
        f"| Flagged | {at['flag_rate']:.1%} ({at['tp'] + at['fp']} patients) | {dv['at_threshold']['flag_rate']:.1%} |",
        "",
        f"Holdout confusion matrix at the threshold: TP {at['tp']}, FP {at['fp']}, FN {at['fn']}, TN {at['tn']}.",
        "",
        "Figures: `reports/figures/`.",
    ]
    (ROOT / "reports" / "results.md").write_text("\n".join(lines) + "\n")


def write_all(results: dict, oof: pd.DataFrame, calibrated: dict, cohort: pd.DataFrame):
    _style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_pr_curves(oof, oof["y"].mean())
    per_fold = pd.DataFrame(results["cv"]["fold_metrics"]).pivot(index="fold", columns="model", values="pr_auc")
    fig_fold_pr_auc(per_fold)
    fig_ablation(pd.DataFrame(results["ablation"]["per_fold"]), pd.DataFrame(results["ablation"]["summary"]))
    fig_calibration(oof, calibrated, results["adopted"])
    fig_threshold(pd.DataFrame(results["threshold_sweep"]), results["run"]["target_recall"])
    write_markdown(results)


if __name__ == "__main__":
    # Re-draw figures and results.md from a finished run without retraining
    import json

    import models as m

    results = json.loads((ROOT / "reports" / "metrics.json").read_text())
    oof = pd.read_csv(ROOT / "reports" / "oof_predictions.csv")
    calibrated = {name: m.crossfit_calibrate(oof[name].values, oof["y"].values) for name in LABELS}
    write_all(results, oof, calibrated, None)
    print("figures and results.md rewritten")
