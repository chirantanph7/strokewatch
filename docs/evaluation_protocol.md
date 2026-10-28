# StrokeWatch evaluation protocol

This is the protocol every model in StrokeWatch is judged by. `train.py` implements it in this order, and `evaluate.py` is the only place metrics are computed.

## 1. Data and splits (U3.1, M1)
- Data: M1's typed table `data/stroke_typed.parquet`, 5,110 records, 249 strokes (4.87%).
- Splits: M1's frozen `data/splits_REAL.json` (seed 42). A 20% stratified **holdout** of 1,022 records, and 5 stratified folds over the 4,088-record **development set**. Each validation fold's prevalence is within 0.5 percentage points of the overall rate, and a test checks this.
- The holdout is locked in code. `dataset.holdout_frame()` raises an error unless it's called with `unlock=True`, and only the last stage of `train.py` does that. Every fit calls `assert_no_holdout()` first.

## 2. Leakage rules
- Every model is one pipeline: `CleanTransformer → feature transformer → estimator`. The whole pipeline is fit on training data, so BMI imputation, scaling and encoding are re-fit inside every training fold and every inner tuning split.
- Hyperparameters are tuned with an inner 3-fold stratified CV **inside** each outer training fold. Outer validation folds are only scored.
- No target encoding, and no statistic is computed from validation or holdout rows.

## 3. Metrics (U3.3, U3.4)
- **Primary:** PR-AUC (average precision). Chance level equals prevalence, about 0.049.
- **Secondary:** ROC-AUC, Brier score (on calibrated probabilities), and recall, precision and specificity at the chosen threshold.
- **Uncertainty:** stratified bootstrap 95% CIs over patients, with 1,000 resamples and seed 42.
- Accuracy is never reported.

## 4. Feature ablation and selection (U2.7)
- Five feature sets go through the same tuned logistic regression on the same 5 folds: `raw`, `+bands` (age bands and BMI categories), `+glucose_bands`, `+comorbidity` (comorbidity count and age × hypertension), and `all_engineered`.
- **Selection rule:** a derived group goes into the final feature set only if adding it to `raw` raises mean fold PR-AUC **and** improves it on at least 3 of the 5 folds.
- An XGBoost with fixed settings is shown alongside, for context only. It doesn't affect selection.

## 5. Candidate models (U3.2, U4.1–U4.3)
| Model | Imbalance handling | Search (inner 3-fold CV, scored by PR-AUC) |
|---|---|---|
| Logistic regression (baseline) | `class_weight="balanced"` | grid over C, 8 values |
| XGBoost (primary candidate) | `scale_pos_weight` = negatives / positives in that training fold | randomized, 60 configurations, seeded |
| Random forest (comparator) | `class_weight="balanced_subsample"` | randomized, 20 configurations, seeded |

Every configuration tried is logged to `reports/search_log.csv`.

## 6. Adoption rule (U4.4, U4.6)
Quoted verbatim from proposal slide 19:

> *Adopt only if PR-AUC gain CI excludes 0 and calibration not worse; else keep baseline.*

How it's applied:
- **PR-AUC gain CI** is a patient-level paired bootstrap of (candidate − baseline) PR-AUC on the pooled out-of-fold predictions. Each resample scores both models on the same patients.
- **Calibration not worse** means the candidate's Brier score is no higher than the baseline's. Both are on out-of-fold probabilities calibrated by cross-fitted Platt scaling, so neither model's calibrator is fit and scored on the same patients.
- If several candidates pass, the one with the largest gain is adopted. If none pass, the baseline is kept.

## 7. Calibration and threshold (U4.5, U3.5)
- Class weighting deliberately inflates scores, so the adopted model's raw scores go through a Platt (logistic) calibrator fit on its development out-of-fold predictions.
- **Threshold policy:** the highest calibrated-risk threshold whose development out-of-fold recall is at least **80%**. Among thresholds that reach that recall, this one flags the fewest patients. The 80% target is provisional and team-chosen, not clinically validated.
- A trade-off table for 50–95% recall is reported so the choice is visible.

## 8. Final model and holdout
- The adopted model is re-tuned and refit on the full development set. The calibrator and threshold from step 7 carry over unchanged.
- The holdout is scored **once**, and all metrics are reported with CIs. No decision is changed after looking at holdout results.

## 9. Deviations from the Part 1 plan
| Planned | What was done | Why |
|---|---|---|
| 5×3 repeated stratified CV | M1's 5 frozen folds, once, plus a patient-level bootstrap | Repeating would mean regenerating M1's frozen splits. The bootstrap gives the uncertainty estimate instead. |
| XGBoost early stopping on an inner split | `n_estimators` tuned inside the randomized search | Simpler inside `RandomizedSearchCV`, and still fit on training folds only. |
| Optional SMOTE experiment (U3.6) | Not run | Optional in the plan, and class weighting is the primary method. |
| Feature selection "kept only if ablation supports" | Made concrete as the 3-of-5-folds rule in section 4 | The Part 1 slide gave no exact rule. This one was written during the Part 3 build, after a smoke run had already shown the ablation numbers, so it is **not** pre-registered. The gains it accepts are small (see results). |
| Separate owners for M3, M4 and M5 | One shared codebase after the team shrank to three | Team change on 26 Oct 2026. |
