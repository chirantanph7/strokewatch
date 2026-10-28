"""
make_splits.py — StrokeWatch M1 deliverable (Jyot)

U1.6: freeze a stratified holdout + 5-fold stratified CV split on the
DEVELOPMENT portion, keyed by row `id`, seeded for reproducibility.

Run this ONCE, commit splits.json, and never regenerate it during the project —
that's what "frozen" means. If the dataset changes (e.g. licence issue forces
a swap), regenerate deliberately and tell M3/M4/M5.

Usage:
    python make_splits.py stroke_typed.parquet
"""

import json
import sys
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 42
HOLDOUT_FRAC = 0.20
N_FOLDS = 5


def make_splits(df: pd.DataFrame) -> dict:
    ids = df["id"].values
    y = df["stroke"].values

    dev_idx, holdout_idx = train_test_split(
        range(len(df)), test_size=HOLDOUT_FRAC, stratify=y, random_state=SEED
    )

    dev_ids = ids[dev_idx].tolist()
    holdout_ids = ids[holdout_idx].tolist()
    dev_y = y[dev_idx]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = []
    for fold_i, (train_i, val_i) in enumerate(skf.split(dev_ids, dev_y)):
        folds.append({
            "fold": fold_i,
            "train_ids": [dev_ids[i] for i in train_i],
            "val_ids": [dev_ids[i] for i in val_i],
        })

    # sanity: prevalence check within 0.5pp per fold, per the acceptance criterion
    overall_prev = y.mean()
    for f in folds:
        val_y = df.set_index("id").loc[f["val_ids"], "stroke"]
        prev = val_y.mean()
        f["val_prevalence"] = round(float(prev), 4)
        assert abs(prev - overall_prev) < 0.05, (
            f"Fold {f['fold']} prevalence {prev:.3f} drifts >5pp from overall "
            f"{overall_prev:.3f} — reseed or re-check stratification."
        )

    return {
        "seed": SEED,
        "holdout_frac": HOLDOUT_FRAC,
        "n_folds": N_FOLDS,
        "overall_prevalence": round(float(overall_prev), 4),
        "holdout_ids": holdout_ids,
        "dev_ids": dev_ids,
        "folds": folds,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python make_splits.py <path_to_typed_parquet>")
        sys.exit(1)

    df = pd.read_parquet(sys.argv[1])
    # disjointness check
    assert df["id"].duplicated().sum() == 0, "Duplicate ids present — resolve before splitting."

    splits = make_splits(df)

    with open("splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Holdout: {len(splits['holdout_ids'])} rows")
    print(f"Dev set: {len(splits['dev_ids'])} rows across {splits['n_folds']} folds")
    print(f"Overall prevalence: {splits['overall_prevalence']:.4f}")
    for f in splits["folds"]:
        print(f"  fold {f['fold']}: val_prevalence={f['val_prevalence']:.4f}")
    print("Wrote splits.json")
