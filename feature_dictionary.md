# StrokeWatch feature dictionary

Auto-generated from `features.py` by `generate_docs.py` -- do not hand-edit.
Regenerate after any change to the transformer. 36 output columns.
Describes the adopted model (logreg), feature config {'bands': True, 'glucose_bands': True, 'comorbidity': True}, fit on the development set.

| Output column | Source column | Transform | Rationale |
|---|---|---|---|
| `gender_Female` | `gender` | one-hot encoded | Demographic risk factor, documented in stroke risk literature. |
| `gender_Male` | `gender` | one-hot encoded | Demographic risk factor, documented in stroke risk literature. |
| `gender_Other` | `gender` | one-hot encoded | Demographic risk factor, documented in stroke risk literature. |
| `work_type_Govt_job` | `work_type` | one-hot encoded | Proxy for activity level and stress exposure. |
| `work_type_Never_worked` | `work_type` | one-hot encoded | Proxy for activity level and stress exposure. |
| `work_type_Private` | `work_type` | one-hot encoded | Proxy for activity level and stress exposure. |
| `work_type_Self-employed` | `work_type` | one-hot encoded | Proxy for activity level and stress exposure. |
| `work_type_children` | `work_type` | one-hot encoded | Proxy for activity level and stress exposure. |
| `Residence_type_Rural` | `Residence_type` | one-hot encoded | Proxy for healthcare access. |
| `Residence_type_Urban` | `Residence_type` | one-hot encoded | Proxy for healthcare access. |
| `smoking_status_Unknown` | `smoking_status` | one-hot encoded | Established modifiable risk factor; Unknown kept as informative missingness, not imputed. |
| `smoking_status_formerly smoked` | `smoking_status` | one-hot encoded | Established modifiable risk factor; Unknown kept as informative missingness, not imputed. |
| `smoking_status_never smoked` | `smoking_status` | one-hot encoded | Established modifiable risk factor; Unknown kept as informative missingness, not imputed. |
| `smoking_status_smokes` | `smoking_status` | one-hot encoded | Established modifiable risk factor; Unknown kept as informative missingness, not imputed. |
| `ever_married_No` | `ever_married` | one-hot encoded | Included for completeness; weak prior evidence of direct causal link. |
| `ever_married_Yes` | `ever_married` | one-hot encoded | Included for completeness; weak prior evidence of direct causal link. |
| `age_band_adult` | `age` | derived: age band (child <18, adult 18-39, middle_age 40-59, senior 60+), one-hot encoded | Coarse, readable age grouping for SHAP explanations. |
| `age_band_child` | `age` | derived: age band (child <18, adult 18-39, middle_age 40-59, senior 60+), one-hot encoded | Coarse, readable age grouping for SHAP explanations. |
| `age_band_middle_age` | `age` | derived: age band (child <18, adult 18-39, middle_age 40-59, senior 60+), one-hot encoded | Coarse, readable age grouping for SHAP explanations. |
| `age_band_senior` | `age` | derived: age band (child <18, adult 18-39, middle_age 40-59, senior 60+), one-hot encoded | Coarse, readable age grouping for SHAP explanations. |
| `bmi_category_normal` | `bmi` | derived: WHO adult BMI category (underweight/normal/overweight/obese), one-hot encoded | WHO-standard BMI categories; clinically interpretable, not diagnostic. |
| `bmi_category_obese` | `bmi` | derived: WHO adult BMI category (underweight/normal/overweight/obese), one-hot encoded | WHO-standard BMI categories; clinically interpretable, not diagnostic. |
| `bmi_category_overweight` | `bmi` | derived: WHO adult BMI category (underweight/normal/overweight/obese), one-hot encoded | WHO-standard BMI categories; clinically interpretable, not diagnostic. |
| `bmi_category_underweight` | `bmi` | derived: WHO adult BMI category (underweight/normal/overweight/obese), one-hot encoded | WHO-standard BMI categories; clinically interpretable, not diagnostic. |
| `glucose_band_elevated` | `avg_glucose_level` | derived: glucose band (normal <100, elevated 100-139, high 140-199, very_high 200+ mg/dL), one-hot encoded | Heuristic, NON-DIAGNOSTIC bands: the dataset does not say whether glucose is fasting or random. |
| `glucose_band_high` | `avg_glucose_level` | derived: glucose band (normal <100, elevated 100-139, high 140-199, very_high 200+ mg/dL), one-hot encoded | Heuristic, NON-DIAGNOSTIC bands: the dataset does not say whether glucose is fasting or random. |
| `glucose_band_normal` | `avg_glucose_level` | derived: glucose band (normal <100, elevated 100-139, high 140-199, very_high 200+ mg/dL), one-hot encoded | Heuristic, NON-DIAGNOSTIC bands: the dataset does not say whether glucose is fasting or random. |
| `glucose_band_very_high` | `avg_glucose_level` | derived: glucose band (normal <100, elevated 100-139, high 140-199, very_high 200+ mg/dL), one-hot encoded | Heuristic, NON-DIAGNOSTIC bands: the dataset does not say whether glucose is fasting or random. |
| `age` | `age` | standard-scaled (fit on training fold only) | Established stroke risk factor; strongest single predictor in literature. |
| `avg_glucose_level` | `avg_glucose_level` | standard-scaled (fit on training fold only) | Established modifiable risk factor (proxy for glycemic control). |
| `bmi` | `bmi` | M1-imputed (train-fold median) then standard-scaled (fit on training fold only) | Established modifiable risk factor. |
| `comorbidity_count` | `hypertension, heart_disease` | derived: hypertension + heart_disease (0-2), then standard-scaled | Combined burden of the two recorded cardiovascular comorbidities. |
| `age_x_hypertension` | `age, hypertension` | derived: age in years x hypertension (0 if no hypertension), then standard-scaled | Lets a linear model give hypertension more weight at older ages. |
| `hypertension` | `hypertension` | passthrough binary, no transform | Established modifiable risk factor. |
| `heart_disease` | `heart_disease` | passthrough binary, no transform | Established comorbidity risk factor. |
| `bmi_missing` | `bmi` | passthrough binary indicator from M1's CleanTransformer | Missingness kept as a signal per M1's schema notes, not discarded on impute. |

Which derived groups are in the model is decided by the U2.7 ablation in `train.py`;
see `reports/results.md` section 2.
