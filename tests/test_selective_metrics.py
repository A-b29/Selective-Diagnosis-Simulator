"""
test_selective_metrics.py - Unit tests for coverage vs selective risk calculations.
"""

import numpy as np
import pytest
from src.selective_metrics import (
    compute_selective_metrics_at_threshold,
    sweep_coverage_risk_curve,
    select_threshold_for_risk_budget
)


def test_selective_metrics_hand_checked():
    # 4 samples: confidences = [0.95, 0.85, 0.75, 0.60]
    # truth = [1, 1, 0, 1], pred = [1, 1, 1, 0] -> sample 3 and 4 are wrong
    y_true = np.array([1, 1, 0, 1])
    y_pred = np.array([1, 1, 1, 0])
    confs = np.array([0.95, 0.85, 0.75, 0.60])

    # Threshold 0.80 -> accepts samples 1 and 2 (both correct!)
    m = compute_selective_metrics_at_threshold(y_true, y_pred, confs, threshold=0.80)
    assert m["accepted_count"] == 2
    assert m["abstained_count"] == 2
    assert m["coverage"] == 0.50
    assert m["selective_risk"] == 0.0  # Zero errors among accepted


def test_zero_accepted_samples_edge_case():
    y_true = np.array([1, 0])
    y_pred = np.array([1, 0])
    confs = np.array([0.70, 0.80])

    # Threshold 0.95 -> 0 samples accepted
    m = compute_selective_metrics_at_threshold(y_true, y_pred, confs, threshold=0.95)
    assert m["accepted_count"] == 0
    assert m["abstained_count"] == 2
    assert m["coverage"] == 0.0
    assert m["selective_risk"] == 0.0


def test_risk_budget_threshold_optimizer():
    y_true = np.array([1, 1, 1, 0])
    y_pred = np.array([1, 1, 0, 0])  # sample 3 is error
    confs = np.array([0.95, 0.90, 0.65, 0.85])

    opt = select_threshold_for_risk_budget(y_true, y_pred, confs, target_risk_budget=0.0)
    # Target risk 0.0 -> threshold must exclude sample 3 (conf 0.65)
    assert opt["optimal_threshold"] > 0.65
    assert opt["val_selective_risk"] == 0.0
