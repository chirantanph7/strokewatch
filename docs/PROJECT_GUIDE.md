# StrokeWatch: the whole project, explained

A walkthrough of every part of StrokeWatch: what each piece does, why it was built that way, and what the numbers mean. Read it top to bottom once and you should be able to explain any slide and answer most questions about the code.

All numbers come from `reports/metrics.json` (the run of 28 Oct 2026). If you re-run `train.py` they will match exactly, because every seed is fixed.

---

## 1. The problem in one paragraph

We have 5,110 patient records, each with 10 risk factors (age, gender, hypertension, heart disease, marital status, work type, residence, average glucose, BMI, smoking) and a label saying whether the patient has had a stroke. 249 have (4.87%). The goal is a model that **ranks** records by stroke risk, gives a **calibrated probability**, **flags** the riskiest for review and **explains** each score, without fooling ourselves about how good it is.

The last part is the hard bit. With only 249 positives, it's easy to get impressive-looking but meaningless numbers. Leakage, tuning on the test set and picking whichever model looks best are all ways to do that. Most of the engineering in StrokeWatch exists to make those mistakes impossible.

## 2. Why accuracy is banned

If a model predicts "no stroke" for everyone, it is right 95.1% of the time and useless. So we never report accuracy, and `evaluate.py` doesn't even have an accuracy function (a test checks this).

**PR-AUC (average precision)** is the primary metric instead. Rank all patients by score, walk down the list, and at each stroke found, note what fraction of patients so far were strokes. Average those fractions. A random ranking gets PR-AUC equal to prevalence (0.049 here), so 0.049 is the real baseline.

**ROC-AUC** is the probability that a random stroke patient is ranked above a random non-stroke patient. It looks flattering on imbalanced data, because there are so many easy negatives, so it's secondary.

**Brier score** is the mean squared error of the probabilities, so it measures calibration and sharpness together. Always predicting the prevalence scores 0.047. Lower is better.

**Recall / precision at a threshold**: of all strokes, what fraction we flag; of all flags, what fraction are strokes.

## 3. Data (M1, Jyot)

Files: `m1_jyot/preprocess.py` (`load_and_type`, `CleanTransformer`), `m1_jyot/make_splits.py`, `data/schema.yaml`.

- **Typing and flags.** Every column gets the dtype in `schema.yaml`. Anomalies (duplicate ids, out-of-range values, the single `gender = Other` row, children with adult ages) are written to a `_flags` column. Nothing is dropped. Three rows end up flagged.
- **Missing BMI (201 rows).** Dropping them would lose stroke patients, and missingness is informative: 19.9% of patients with missing BMI had a stroke, against 4.3% of the rest. So `CleanTransformer` adds `bmi_missing` (0/1) and *then* fills BMI with the median. The median is learned in `fit()` from training rows only.
- **`smoking_status = Unknown` (1,544 rows)** is kept as its own category rather than treated as missing, because "unknown" is not random.
- **Frozen splits.** A 20% stratified holdout (1,022 patients, 50 strokes), then 5 stratified folds over the remaining 4,088 development patients. Seed 42, keyed by patient id. Every validation fold has prevalence within 0.5 points of 4.87% (tested).
- **Reproducibility.** `make_data.py` checks the CSV's SHA-256 and rebuilds the typed table and splits. We verified that it reproduces them exactly.

## 4. Features (M2, Chirantan)

File: `features.py`. Output: 36 columns (`feature_names.json`, `feature_dictionary.md`).

`build_feature_transformer(bands, glucose_bands, comorbidity)` returns a two-step pipeline:

1. **`BandFeatures`** is stateless. It drops `id` and `_flags`, then adds the derived columns that are switched on:
   - `bands` (U2.1): `age_band` (child <18, adult 18–39, middle_age 40–59, senior 60+) and `bmi_category` (WHO adult cut-offs 18.5 / 25 / 30).
   - `glucose_bands` (U2.2): normal <100, elevated 100–139, high 140–199, very_high ≥200 mg/dL. These are **heuristic and non-diagnostic**, because we don't know whether readings were fasting.
   - `comorbidity` (U2.3): `comorbidity_count` = hypertension + heart_disease, and `age_x_hypertension` = age × hypertension.
2. **`ColumnTransformer`** one-hot encodes the categoricals (with `handle_unknown="ignore"`, so the rare `gender = Other` can't crash a fold that never saw it), standard-scales the numerics, and passes the binary flags through.

"Stateless" matters: the bin edges are fixed clinical numbers, not learned from data, so `BandFeatures` can't leak. The scaler is the only stateful part, and it's fit inside the training fold.

**Leakage tests** (`tests/test_leakage.py`): the scaler mean equals the training fold's mean (not the combined data's), the BMI median equals the training median, and pushing extreme "poisoned" validation data through `transform()` doesn't move any fitted statistic.

### The ablation (U2.7), and how to talk about it

The same tuned logistic regression was run on the same 5 folds with five feature sets:

| Config | Columns | Mean PR-AUC | Change vs raw | Folds improved |
|---|---|---|---|---|
| raw | 21 | 0.2227 | n/a | n/a |
| + age/BMI bands | 29 | 0.2236 | +0.0009 | 3/5 |
| + glucose bands | 25 | 0.2302 | +0.0075 | 5/5 |
| + comorbidity | 23 | 0.2247 | +0.0020 | 3/5 |
| all engineered | 35 | 0.2274 | +0.0048 | n/a |

The rule (keep a group if it raises the mean **and** improves at least 3 of 5 folds) kept all three. **But the honest reading is that the gains are tiny.** The fold-to-fold standard deviation is 0.044, so a +0.001 gain is noise. Only glucose bands helped on every fold.

Why so little? Logistic regression with raw age already captures most of the signal. Bands mostly re-express information that's already there. That's a legitimate finding, and saying so clearly is stronger than overselling it. Also be upfront that this rule was written during the build, after a smoke run, so it is not pre-registered.

## 5. Evaluation protocol and models (M3/M4, Arth)

Files: `evaluate.py`, `models.py`, `train.py`, `docs/evaluation_protocol.md`.

### Nested cross-validation, in plain words
- **Outer loop:** the 5 frozen folds. Each time, 4 folds train and 1 validates. Every development patient is predicted exactly once by a model that never saw them. These are the **out-of-fold (OOF) predictions**.
- **Inner loop:** inside each outer training set, a 3-fold CV picks hyperparameters. The outer validation fold is never used for tuning.
- Because the *whole pipeline* (clean → features → model) is what gets fit, imputation and scaling are re-learned in every inner and outer training split.

### The three models
| Model | Handling imbalance | Tuning | Pooled OOF PR-AUC |
|---|---|---|---|
| Logistic regression (baseline) | `class_weight="balanced"` | C over 8 values (chose C = 0.1 on the full dev set) | **0.215** (0.180–0.264) |
| XGBoost | `scale_pos_weight` = negatives / positives in the training fold (~19.5) | 60 random configs | 0.186 (0.154–0.231) |
| Random forest | `class_weight="balanced_subsample"` | 20 random configs | 0.176 (0.146–0.223) |

Note that XGBoost's search kept picking **depth-2 trees with a small learning rate**, which is close to an additive, nearly linear model. That's a hint the data has little interaction structure for trees to exploit.

### The adoption rule
Written in Part 1, before any results: *"Adopt only if PR-AUC gain CI excludes 0 and calibration not worse; else keep baseline."*

- The CI comes from a **paired bootstrap**: resample patients 1,000 times, and each time compute PR-AUC for *both* models on the *same* resample, then take the difference. Pairing cancels out "this resample happened to be easy" noise, so the CI is about the difference between the models.
- XGBoost: −0.029 (−0.064 to +0.003), so the CI includes 0 and it **fails**. Random forest: −0.039 (−0.076 to −0.002), which is *significantly worse* and **fails**.
- So logistic regression is kept. That's a defensible, pre-specified decision, and it's the most important "maturity signal" in the project.

### Calibration
`class_weight="balanced"` tells the model to treat strokes as if they were about 50% of the data, so its raw scores are inflated: a raw 0.85 meant roughly a 21% stroke rate. A **Platt calibrator**, a one-feature logistic regression on the logit of the raw score, is fit on the dev OOF predictions and maps scores back to honest probabilities. It is monotone, so it never changes the ranking (PR-AUC and ROC-AUC are unchanged). It only changes the scale.

For the Brier comparison between models, each model's OOF scores were calibrated with **cross-fitting** (5-fold within the OOF set). That way no calibrator is fit and scored on the same patients.

### Threshold
The policy: the **highest** threshold that still catches at least 80% of strokes on dev OOF predictions. It's the highest because that flags the fewest people among thresholds meeting the target. Result: **0.058 calibrated risk**.

| Recall target | Precision | Flagged |
|---|---|---|
| 50% | 0.185 | 13% |
| 70% | 0.172 | 20% |
| **80%** | **0.143** | **27%** |
| 95% | 0.099 | 47% |

The 80% target is the team's provisional choice. In reality a clinician would set it based on the cost of a missed stroke versus a follow-up.

### The holdout
Opened once. The final model is retuned and refit on all 4,088 dev patients, and the calibrator and threshold are carried over unchanged.

- PR-AUC **0.257** (0.177–0.370), ROC-AUC 0.837, Brier 0.042
- At the threshold: **40 of 50 strokes caught**, 230 false alarms, 10 missed, 742 correctly not flagged. That's recall 0.80 and precision 0.148, with about 7 reviews per stroke found.
- The holdout PR-AUC is a bit *higher* than dev (0.215), but its CI is wide (only 50 strokes) and contains the dev value. Don't claim it "generalises better". Say it's consistent.

## 6. Explainability (M5a, Chirantan)

File: `explain.py`. Output: `reports/shap_global.csv`, `reports/figures/shap_global.png`.

- **SHAP** splits one prediction into a contribution per input, so that base value + Σ contributions = the model's output. Here the output is the log-odds, before calibration. For logistic regression this is exact: contribution = coefficient × (value − background mean), via `shap.LinearExplainer` with a 200-patient dev background sample.
- **Grouping:** the model sees 36 columns, but people think in 13 variables. Because SHAP values add, the one-hot columns (`smoking_status_smokes`, …) and bands (`age_band_senior`) are summed back to their source (`Smoking status`, `Age`). A test checks the totals are unchanged.
- **Additivity test:** base + Σ SHAP = `decision_function(x)` to 1e-4, for 100 records.
- **Global importance** (mean |SHAP|, log-odds): Age 1.85, Comorbidity count 0.17, Glucose 0.16, Ever married 0.15, Smoking 0.13, BMI missing 0.13, Work type 0.13, …

**How to read it, and the traps:**
1. Age dominates. Stroke patients' median age is 71, against 43 for everyone else.
2. **Correlated inputs share credit.** Hypertension shows up three times (the flag, the comorbidity count, age × hypertension), so each looks small. Read them together. In record 1001 (72, hypertensive), age × hypertension even *lowers* the score: the model learned that hypertension adds less extra risk at older ages, once age is already counted.
3. **Proxies.** "Ever married" ranks high because married people are older (median age 54 vs 18). It isn't a cause.
4. SHAP explains **the model**, not biology. Every explanation in the app carries that disclaimer.

## 7. Dashboard (M5b, Jyot)

Files: `app/streamlit_app.py`, `app/validation.py`, `app/sample_records.csv` (6 synthetic patients).

- **Score records:** upload a CSV or use the synthetic sample. `validate_records()` checks it against `schema.yaml`. Hard errors (missing column, unknown category, non-number) stop scoring with a readable message. Soft warnings (out of range, missing BMI) are shown but still scored, mirroring M1's "flag, don't drop". The output is risk, flag and top 3 reasons, downloadable as CSV.
- **Record inspector:** one record's inputs, calibrated risk, flag and top 5 SHAP contributions as a diverging bar chart.
- **Cohort view:** score distributions for stroke vs non-stroke (as a share of each group, so the 5% group is visible) with the threshold line, plus the PR curve.
- **Model card:** intended use, holdout metrics and limitations.
- A **non-clinical notice** sits above every tab. `tests/test_explain_app.py` runs the app headlessly and checks the notice and all four tabs render.

## 8. Engineering and reproducibility

- `make all` runs data → train → explain → docs → test. The full training run takes about 7 minutes on a laptop.
- **36 tests**: leakage (7), protocol (10), metrics checked against hand calculations (9), explainability and app (10).
- `reports/results.md` and the figures are generated by code, and so is every number in the deck.
- Raw data isn't in the repo (licence). `make data` rebuilds it from the Kaggle CSV after a checksum check.

## 9. Deviations from the Part 1 plan (be ready to explain)

| Planned | Done | Why |
|---|---|---|
| 5×3 repeated CV | The 5 frozen folds once, plus bootstrap CIs | Repeating would mean regenerating M1's frozen splits |
| XGBoost early stopping | `n_estimators` tuned in the search | Simpler inside `RandomizedSearchCV`, still in-fold |
| SMOTE experiment (optional) | Not run | Class weighting was the primary method |
| "Keep a feature if ablation supports it" | The 3-of-5-folds rule | Needed a concrete rule; it was written after a smoke run |

## 10. Limitations (say them before you're asked)

One public dataset of unknown origin and no external validation. Cross-sectional labels with no time horizon. Missing clinical variables (blood pressure values, AF, cholesterol, medication). Low precision is inherent at ~5% prevalence. Wide CIs (50 holdout strokes). Glucose meaning is unknown. Small subgroups (gender = Other n = 1, children) are barely estimated.

## 11. Glossary

- **Leakage:** information from validation or test data influencing training, e.g. imputing with a median computed on all rows. It makes results look better than they are.
- **Stratified split:** each split keeps the same stroke percentage.
- **Out-of-fold prediction:** a prediction for a patient made by a model trained without them.
- **Bootstrap CI:** resample patients with replacement many times, recompute the metric, and take the 2.5th and 97.5th percentiles.
- **Calibration:** whether "20% risk" really means about 20% of such patients had a stroke.
- **Log-odds:** log(p / (1 − p)). Logistic regression is linear in log-odds, which is why SHAP is exact for it.
