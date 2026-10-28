"""
StrokeWatch -- access to M1's typed data and frozen splits.

Every model, report and test reads data through this module, so there is one
place that enforces the protocol (U3.1):
  - the 5 frozen folds come from M1's splits file and are never regenerated here;
  - the 20% holdout is locked. `holdout_frame()` refuses unless the caller
    passes `unlock=True`, which only the final evaluation step in train.py does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "stroke_typed.parquet"
SPLITS_PATH = ROOT / "data" / "splits_REAL.json"
TARGET = "stroke"


class HoldoutLockedError(RuntimeError):
    pass


def load_typed() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Download the Kaggle CSV and run `make data` (see README)."
        )
    return pd.read_parquet(DATA_PATH)


def load_splits() -> dict:
    with open(SPLITS_PATH) as f:
        return json.load(f)


def _rows(df: pd.DataFrame, ids) -> tuple[pd.DataFrame, pd.Series]:
    """Rows for `ids`, in the order the splits file lists them."""
    part = df.set_index("id").loc[list(ids)].reset_index()
    return part.drop(columns=[TARGET]), part[TARGET].astype(int)


def dev_frame(df: pd.DataFrame, splits: dict) -> tuple[pd.DataFrame, pd.Series]:
    return _rows(df, splits["dev_ids"])


def fold_frames(df: pd.DataFrame, splits: dict, fold: int):
    """(X_train, y_train, X_val, y_val) for one frozen fold of the dev set."""
    f = splits["folds"][fold]
    X_tr, y_tr = _rows(df, f["train_ids"])
    X_va, y_va = _rows(df, f["val_ids"])
    return X_tr, y_tr, X_va, y_va


def holdout_frame(df: pd.DataFrame, splits: dict, unlock: bool = False):
    if not unlock:
        raise HoldoutLockedError(
            "The holdout is locked. Only the final evaluation in train.py may read it."
        )
    return _rows(df, splits["holdout_ids"])


def assert_no_holdout(ids, splits: dict) -> None:
    """Guard used before every fit: no holdout id may enter training or threshold code."""
    leaked = set(ids) & set(splits["holdout_ids"])
    if leaked:
        raise AssertionError(f"{len(leaked)} holdout ids reached a fit/threshold step")
