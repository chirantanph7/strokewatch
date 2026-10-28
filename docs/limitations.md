### Limitations

- **Not a clinical tool.** StrokeWatch is a course research prototype. It has no regulatory status and has never been tested in a clinical setting.
- **One small, public dataset.** 5,110 records with 249 strokes, from a Kaggle dataset whose original source isn't disclosed. With no external validation cohort, performance on any other population is unknown.
- **Retrospective, cross-sectional labels.** `stroke` records whether a patient has had a stroke, not whether they will have one. The model finds risk-factor profiles that go with a recorded stroke. It does not predict future events over a time horizon.
- **Missing clinical variables.** There's no blood pressure reading, cholesterol, atrial fibrillation, medication or family history. `hypertension` and `heart_disease` are yes/no flags only.
- **Uncertain glucose meaning.** The dataset doesn't say whether `avg_glucose_level` was fasting or random, so glucose bands are heuristic and non-diagnostic.
- **Low precision at useful recall.** At ~5% prevalence, most flagged records are not strokes. A flag means "worth a closer look in a research setting", nothing more.
- **Explanations are not causes.** SHAP values describe how the model uses its inputs. They show association in this dataset, not cause, and they're not clinical reasoning.
- **Possible bias.** Subgroups such as `gender = Other` (n=1), children and some work types are small, so the model's behaviour in those groups is poorly estimated.
