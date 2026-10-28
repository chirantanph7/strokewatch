# StrokeWatch: project context for Claude Code

Group 8 course project. A stroke-risk classifier built on the public Kaggle stroke dataset
(5,110 rows, 249 strokes, ~4.9% prevalence). It is a research prototype, not a clinical tool.

This file is the handoff from earlier work in Claude chat. Read it before changing anything.

## Team (changed 26 Oct 2026: five members down to three)

Dhruv (advanced model) and Aaditya (explainability and dashboard) left the group.
The redistribution below is what the final deck (presentation/StrokeWatch_Final.pptx) presents.
Confirm it with Jyot and Arth before the final presentation (20-21 Nov 2026).

| Member | Original role | Proposed extra work |
|---|---|---|
| Chirantan (owner of this repo) | M2 feature engineering | SHAP explainability (from Aaditya) |
| Jyot | M1 data cleaning, done and delivered | Dashboard (from Aaditya) |
| Arth | M3 baseline model and `evaluate.py` | Advanced model (from Dhruv) |

The scope cuts proposed on 26 Oct were not needed: every unit was built (28 Oct), including random
forest, the cohort view, glucose bands and comorbidity features. Only the optional SMOTE run (U3.6) was skipped.

## Pipeline order

M1 `CleanTransformer` (Jyot) → M2 `FeatureTransformer` (Chirantan) → models (Arth) → SHAP and dashboard.
Every step is fit on the training fold only. Never fit anything on the full dataset or the holdout.

## Files

| Path | What it is |
|---|---|
| `features.py` | M2 `build_feature_transformer(bands, glucose_bands, comorbidity)`. `python features.py` runs the M1→M2 handoff on fold 0. |
| `dataset.py` | Frozen-split access. `holdout_frame()` raises unless `unlock=True`; `assert_no_holdout()` guards every fit. |
| `evaluate.py` | Shared metrics (no accuracy), bootstrap CIs, threshold policy, paired bootstrap deltas. |
| `models.py` | Pipelines (clean → features → model), search spaces, `PlattCalibrator`, `StrokeModel` (what the app loads). |
| `train.py` / `report.py` | Full experiment: ablation, LR/XGBoost/RF, adoption rule, calibration, threshold, one holdout eval. `report.py` alone redraws figures from a finished run. |
| `explain.py` | SHAP (Linear or Tree explainer), grouping to source variables, global importance figure. |
| `app/` | Streamlit dashboard (`streamlit_app.py`), upload validation against `data/schema.yaml`, synthetic sample records. |
| `tests/` | 36 tests: leakage, protocol, metrics vs hand calculations, SHAP additivity, app smoke test. |
| `generate_docs.py` | Writes `feature_dictionary.md` and `feature_names.json` from the trained model. Never hand-edit those two files. |
| `make_data.py` | Rebuilds `data/` from the raw Kaggle CSV after a SHA-256 check, using Jyot's code. |
| `upstream_preprocess.py` | Copy of Jyot's `preprocess.py` (`CleanTransformer`). Treat as M1's code; don't change it without telling Jyot. |
| `data/` | Jyot's typed parquet, frozen splits (`splits_REAL.json`) and schema. Only `schema.yaml` is in git. |
| `m1_jyot/` | Jyot's original delivery. The raw CSV and splits.json are gitignored (licence note). |
| `reports/` | Generated: `results.md`, `metrics.json`, `figures/`, `shap_global.csv`. |
| `models/` | Generated, gitignored: `final_model.joblib`, OOF and holdout scores, SHAP background. |
| `docs/` | Evaluation protocol, model card, limitations, `PROJECT_GUIDE.md` (full explainer), Gantt. |
| `presentation/` | Final deck (gitignored: it has student IDs). |
| `reference/` | Earlier decks and the course format (gitignored: course material and student IDs). |
| `private/` | Chirantan's resume and viva notes (gitignored). |

## Things that are easy to get wrong

- Output column count depends on the fit data. Fold 0 gives 29 columns; fitting on all 5,110 rows gives 30 because `gender_Other` (one row) only appears then. Always read names from `get_feature_names_out()`, never hard-code a count.
- `CleanTransformer` adds `bmi_missing` before median-imputing BMI. It is a real feature, kept on purpose.
- `id` and `_flags` are dropped inside `BandFeatures`, not upstream, so the audit trail survives.
- `pd.cut(...).astype(str)` keeps missing values as real NaN in pandas 3, so `BandFeatures` uses `fillna("missing")`.
- Accuracy is meaningless at 4.9% prevalence. The Part 1 plan uses PR-AUC as the primary metric.

## Status (28 Oct 2026)

Everything planned is built and run. Headline (locked holdout, evaluated once): logistic regression
adopted; PR-AUC 0.257 (chance 0.049), ROC-AUC 0.837, recall 0.80 / precision 0.148 at threshold 0.058.
XGBoost and random forest failed the pre-registered adoption rule. Feature ablation gains ≤ +0.0075 PR-AUC.

Open:
- Confirm ownership split and AI-use disclosure with the team and course staff
- Get Jyot's OK before publishing `m1_jyot/` publicly
- Email deck + software before the final presentation (20-21 Nov 2026)

## Commands

```
make setup      # .venv + requirements
make data       # needs m1_jyot/healthcare-dataset-stroke-data.csv
make train      # ~7 min
make explain docs test
make app
```

Built and tested with Python 3.14, pandas 3.1, scikit-learn 1.9, numpy 2.5, xgboost 3.4, shap 0.54
(exact pins in requirements-lock.txt). Use `.venv/bin/python`; the system has no `python`.
