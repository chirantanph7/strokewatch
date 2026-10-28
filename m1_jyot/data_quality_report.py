"""
data_quality_report.py — StrokeWatch M1 deliverable (Jyot)

U1.7: regenerate a Markdown data-quality report straight from the raw CSV,
every time, so it can never drift from the actual data.

Usage:
    python data_quality_report.py healthcare-dataset-stroke-data.csv
"""

import sys
from preprocess import load_and_type, file_checksum


def build_report(raw_csv_path: str) -> str:
    checksum = file_checksum(raw_csv_path)
    df = load_and_type(raw_csv_path)

    n = len(df)
    n_pos = int(df["stroke"].sum())
    prevalence = n_pos / n

    missing_by_col = df.isna().sum()
    missing_by_col = missing_by_col[missing_by_col > 0]

    flagged = df[df["_flags"] != ""]
    flag_counts = {}
    for flags in flagged["_flags"]:
        for f in flags.split(";"):
            flag_counts[f] = flag_counts.get(f, 0) + 1

    lines = []
    lines.append("# StrokeWatch — Data Quality Report")
    lines.append("")
    lines.append(f"- Source file checksum (SHA-256): `{checksum}`")
    lines.append(f"- Rows: {n}")
    lines.append(f"- Positive (stroke=1): {n_pos} ({prevalence:.2%})")
    lines.append("")
    lines.append("## Missingness")
    lines.append("")
    if len(missing_by_col) == 0:
        lines.append("No missing values found (unexpected — re-check `na_values` parsing).")
    else:
        lines.append("| Column | Missing count | Missing % |")
        lines.append("|---|---|---|")
        for col, cnt in missing_by_col.items():
            lines.append(f"| {col} | {cnt} | {cnt/n:.2%} |")
    lines.append("")
    lines.append("## Row-level flags (nothing dropped — every flagged row is retained)")
    lines.append("")
    if not flag_counts:
        lines.append("No anomalies flagged.")
    else:
        lines.append("| Flag | Count | Decision |")
        lines.append("|---|---|---|")
        decisions = {
            "duplicate_id": "Investigate manually — should not occur; escalate to team if found.",
            "duplicate_row": "Retain unless confirmed as an ingestion artifact; log evidence either way.",
            "gender_other": "Retained. n is small; category kept as-is, not merged into another class.",
            "age_out_of_range": "Retained, flagged for review — plausible range is a sanity check, not ground truth.",
            "avg_glucose_level_out_of_range": "Retained, flagged for review.",
            "bmi_out_of_range": "Retained, flagged for review.",
            "work_type_age_mismatch": "Retained, flagged — 'children' work_type with adult age needs manual look.",
        }
        for flag, cnt in flag_counts.items():
            lines.append(f"| {flag} | {cnt} | {decisions.get(flag, 'Needs a decision — add to this table.')} |")
    lines.append("")
    lines.append("## Design commitments this report enforces")
    lines.append("")
    lines.append("- No rows are dropped for missingness or range violations — only flagged.")
    lines.append("- BMI missingness is handled downstream via `CleanTransformer`, fit on training folds only.")
    lines.append("- `smoking_status == 'Unknown'` is kept as an informative category, not treated as null.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python data_quality_report.py <path_to_raw_csv>")
        sys.exit(1)

    report = build_report(sys.argv[1])
    with open("data_quality_report.md", "w") as f:
        f.write(report)
    print("Wrote data_quality_report.md")
    print()
    print(report)
