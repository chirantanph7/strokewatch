# StrokeWatch — Data Quality Report

- Source file checksum (SHA-256): `644d473b05d2797006bd94865e4f8bb057f0c721617911613c82c8fcfc707420`
- Rows: 5110
- Positive (stroke=1): 249 (4.87%)

## Missingness

| Column | Missing count | Missing % |
|---|---|---|
| bmi | 201 | 3.93% |

## Row-level flags (nothing dropped — every flagged row is retained)

| Flag | Count | Decision |
|---|---|---|
| bmi_out_of_range | 2 | Retained, flagged for review. |
| gender_other | 1 | Retained. n is small; category kept as-is, not merged into another class. |

## Design commitments this report enforces

- No rows are dropped for missingness or range violations — only flagged.
- BMI missingness is handled downstream via `CleanTransformer`, fit on training folds only.
- `smoking_status == 'Unknown'` is kept as an informative category, not treated as null.
