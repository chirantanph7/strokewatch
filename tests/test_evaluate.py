"""
evaluate.py checked against metrics worked out by hand on toy data
(proposal slide 16: "Unit tests against hand-computed metrics on toy data").
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as ev  # noqa: E402

# 10 patients, 3 strokes. Sorted by score: 1, 0, 1, 0, 0, 1, 0, 0, 0, 0
Y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
P = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1])


def test_pr_auc_matches_hand_computed_average_precision():
    # positives found at ranks 1, 3, 6: precision there 1/1, 2/3, 3/6
    expected = (1 + 2 / 3 + 3 / 6) / 3
    assert ev.pr_auc(Y, P) == pytest.approx(expected)


def test_roc_auc_matches_hand_computed_pair_count():
    # positive/negative pairs ranked correctly: 7 + 6 + 4 = 17 of 21
    assert ev.roc_auc(Y, P) == pytest.approx(17 / 21)


def test_brier_is_mean_squared_error():
    assert ev.brier(Y, P) == pytest.approx(np.mean((P - Y) ** 2))


def test_threshold_metrics_confusion_counts():
    m = ev.threshold_metrics(Y, P, 0.5)  # flags the top 5: 1,0,1,0,0
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 3, 1, 4)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 5)
    assert m["specificity"] == pytest.approx(4 / 7)
    assert m["flag_rate"] == pytest.approx(0.5)


def test_threshold_for_recall_picks_the_highest_threshold_that_meets_target():
    # recall 2/3 needs the 2nd-best positive caught -> threshold 0.7
    assert ev.threshold_for_recall(Y, P, 0.6) == pytest.approx(0.7)
    # recall 1.0 needs all three -> threshold 0.4
    assert ev.threshold_for_recall(Y, P, 1.0) == pytest.approx(0.4)
    for target in [0.3, 0.5, 0.8, 1.0]:
        t = ev.threshold_for_recall(Y, P, target)
        assert ev.threshold_metrics(Y, P, t)["recall"] >= target


def test_constant_score_pr_auc_equals_prevalence():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    assert ev.pr_auc(y, np.full(len(y), 0.5)) == pytest.approx(y.mean())


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.1).astype(int)
    p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    est, lo, hi = ev.bootstrap_ci(y, p, ev.pr_auc, n_boot=200)
    assert lo <= est <= hi
    assert hi - lo > 0


def test_paired_delta_of_identical_models_is_zero():
    d = ev.paired_bootstrap_delta(Y, P, P, n_boot=50)
    assert d["delta"] == 0 and d["ci_low"] == 0 and d["ci_high"] == 0


def test_no_accuracy_metric_is_offered():
    assert not any("accuracy" in name for name in dir(ev))
