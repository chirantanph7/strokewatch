### Model card

**What it does.** Gives a calibrated probability that a tabular patient record is stroke-positive, from 10 demographic and clinical risk factors. Records at or above a fixed threshold are flagged for review.

**Pipeline.** M1 `CleanTransformer` (BMI median imputation plus a missing-indicator) → M2 feature transformer (one-hot encoding, scaling, and the derived features the ablation kept) → the adopted classifier → Platt calibration. Every step is fit on training data only.

**Intended use.** Teaching and research on imbalanced clinical classification, leakage-safe pipelines and explainability, within CS F434.

**Out of scope.** Diagnosis, emergency triage, ranking live patients by urgency, predicting future risk over a time horizon, and any decision about a real person.

**Training data.** Development split (80%, 4,088 records) of the Kaggle stroke dataset (fedesoriano), using M1's frozen seed-42 splits. The remaining 20% (1,022 records) was locked and evaluated once.

**Metrics.** PR-AUC is the primary metric. ROC-AUC, Brier score, and recall and precision at the threshold are also reported, all with bootstrap 95% CIs. Accuracy is not reported because it is meaningless at this prevalence.

**Threshold.** Chosen on development out-of-fold predictions as the highest threshold that still catches 80% of strokes. The 80% target is provisional and was set by the team, not a clinician.
