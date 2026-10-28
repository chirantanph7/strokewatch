"""
StrokeWatch -- model pipelines (M3 baseline, M4 advanced models).

Every model is one sklearn Pipeline: M1 CleanTransformer -> M2 feature
transformer -> estimator. Because the whole pipeline is what gets fit, any
cross-validation search over it refits cleaning and scaling inside each inner
training split too, so tuning cannot leak validation statistics.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from features import build_feature_transformer
from upstream_preprocess import CleanTransformer

SEED = 42
INNER_FOLDS = 3
XGB_BUDGET = 60  # proposal slide 19: fixed budget of 60 configurations
RF_BUDGET = 20

# U3.2: C chosen by inner CV, one value per outer training fold
LOGREG_GRID = {"model__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}

# U4.2: fixed, seeded, laptop-scale search space
XGB_SPACE = {
    "model__n_estimators": randint(100, 600),
    "model__max_depth": randint(2, 6),
    "model__learning_rate": loguniform(0.01, 0.2),
    "model__subsample": uniform(0.6, 0.4),
    "model__colsample_bytree": uniform(0.5, 0.5),
    "model__min_child_weight": randint(1, 20),
    "model__reg_lambda": loguniform(0.1, 20),
    "model__gamma": uniform(0, 2),
}

RF_SPACE = {
    "model__max_depth": [4, 6, 8, 12, None],
    "model__min_samples_leaf": [1, 5, 10, 20, 40],
    "model__max_features": ["sqrt", 0.3, 0.5],
}


def make_pipeline(estimator, feature_config: dict | None = None) -> Pipeline:
    return Pipeline(
        steps=[
            ("clean", CleanTransformer()),
            ("features", build_feature_transformer(**(feature_config or {}))),
            ("model", estimator),
        ]
    )


def _inner_cv():
    return StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)


def logistic_search(feature_config: dict | None = None, n_jobs: int = -1) -> GridSearchCV:
    """M3 baseline: L2 logistic regression, class_weight=balanced."""
    est = LogisticRegression(class_weight="balanced", max_iter=5000)
    return GridSearchCV(
        make_pipeline(est, feature_config),
        LOGREG_GRID,
        scoring="average_precision",
        cv=_inner_cv(),
        n_jobs=n_jobs,
    )


def scale_pos_weight(y) -> float:
    """Negatives per positive in the training fold (U4.1)."""
    y = np.asarray(y)
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def xgb_estimator(y_train, **params) -> XGBClassifier:
    return XGBClassifier(
        scale_pos_weight=scale_pos_weight(y_train),
        eval_metric="aucpr",
        tree_method="hist",
        random_state=SEED,
        n_jobs=1,
        **params,
    )


def xgb_search(y_train, feature_config: dict | None = None, n_iter: int = XGB_BUDGET, n_jobs: int = -1):
    """M4 primary candidate. scale_pos_weight comes from this training fold only."""
    return RandomizedSearchCV(
        make_pipeline(xgb_estimator(y_train), feature_config),
        XGB_SPACE,
        n_iter=n_iter,
        scoring="average_precision",
        cv=_inner_cv(),
        random_state=SEED,
        n_jobs=n_jobs,
    )


def rf_search(feature_config: dict | None = None, n_iter: int = RF_BUDGET, n_jobs: int = -1):
    """M4 comparator: random forest with balanced class weights (U4.3)."""
    est = RandomForestClassifier(
        n_estimators=400, class_weight="balanced_subsample", random_state=SEED, n_jobs=1
    )
    return RandomizedSearchCV(
        make_pipeline(est, feature_config),
        RF_SPACE,
        n_iter=n_iter,
        scoring="average_precision",
        cv=_inner_cv(),
        random_state=SEED,
        n_jobs=n_jobs,
    )


class PlattCalibrator:
    """Maps a model's raw score to a calibrated probability (U4.5).
    Fit on out-of-fold dev predictions, never on the holdout. Class-weighted
    models over-predict risk by design, so their raw scores are not
    probabilities; this logistic map on the score's logit restores that."""

    def fit(self, scores, y):
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(scores), y)
        return self

    def transform(self, scores):
        return self.lr_.predict_proba(self._logit(scores))[:, 1]

    @staticmethod
    def _logit(scores):
        s = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(s / (1 - s)).reshape(-1, 1)


def crossfit_calibrate(scores, y, n_splits: int = 5) -> np.ndarray:
    """Out-of-sample calibrated probabilities for a set of OOF scores, so that
    Brier scores of different models can be compared without fitting and
    scoring the calibrator on the same patients."""
    scores, y = np.asarray(scores), np.asarray(y)
    out = np.empty(len(y))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=SEED).split(scores, y):
        out[te] = PlattCalibrator().fit(scores[tr], y[tr]).transform(scores[te])
    return out


class StrokeModel:
    """The adopted model as one object: fitted pipeline + calibrator + the
    threshold chosen on dev data. This is what the dashboard loads."""

    def __init__(self, name, pipeline, calibrator, threshold, target_recall, feature_config):
        self.name = name
        self.pipeline = pipeline
        self.calibrator = calibrator
        self.threshold = threshold
        self.target_recall = target_recall
        self.feature_config = feature_config

    def raw_score(self, X):
        return self.pipeline.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        return self.calibrator.transform(self.raw_score(X))

    def flag(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def transform_features(self, X):
        return self.pipeline[:-1].transform(X)

    @property
    def feature_names(self) -> list[str]:
        # the ColumnTransformer knows its output names without needing input names
        return list(self.pipeline.named_steps["features"][-1].get_feature_names_out())
