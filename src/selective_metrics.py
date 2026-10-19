"""
selective_metrics.py - Coverage vs. Selective Risk evaluation engine.

Implements selective classification / abstention mechanisms:
- Coverage: fraction of predictions retained
- Selective Risk: empirical error rate among accepted cases
- Zero-accepted cases safe handling
- Optimal threshold selection based on a target risk budget (fit on validation set only)
"""

from typing import Dict, Any, List, Optional
import numpy as np


def compute_selective_metrics_at_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """
    Evaluate model performance when predictions with confidence < threshold are rejected (abstained).
    """
    total_samples = len(y_true)
    if total_samples == 0:
        return {
            "threshold": threshold,
            "coverage": 0.0,
            "selective_risk": 0.0,
            "accepted_count": 0,
            "abstained_count": 0,
            "accepted_errors": 0,
        }

    accepted_mask = (confidences >= threshold)
    accepted_count = int(np.sum(accepted_mask))
    abstained_count = total_samples - accepted_count
    coverage = float(accepted_count / total_samples)

    if accepted_count == 0:
        # Zero accepted: risk is undefined; conventionally set to 0.0 or None
        selective_risk = 0.0
        accepted_errors = 0
    else:
        accepted_errors = int(np.sum(y_true[accepted_mask] != y_pred[accepted_mask]))
        selective_risk = float(accepted_errors / accepted_count)

    return {
        "threshold": float(threshold),
        "coverage": coverage,
        "selective_risk": selective_risk,
        "accepted_count": accepted_count,
        "abstained_count": abstained_count,
        "accepted_errors": accepted_errors,
    }


def sweep_coverage_risk_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    num_thresholds: int = 50
) -> List[Dict[str, Any]]:
    """Generate the full Coverage vs. Selective Risk curve across confidence thresholds."""
    thresholds = np.linspace(0.50, 0.99, num_thresholds)
    curve = []
    for th in thresholds:
        curve.append(compute_selective_metrics_at_threshold(y_true, y_pred, confidences, th))
    return curve


def select_threshold_for_risk_budget(
    val_y_true: np.ndarray,
    val_y_pred: np.ndarray,
    val_confidences: np.ndarray,
    target_risk_budget: float,
    num_thresholds: int = 100
) -> Dict[str, Any]:
    """
    Find the threshold theta* that satisfies Selective Risk <= target_risk_budget
    while maximizing Coverage, fit STRICTLY on validation data.
    """
    thresholds = np.linspace(0.50, 0.999, num_thresholds)
    best_threshold = None
    best_coverage = -1.0
    achieved_val_risk = None

    for th in thresholds:
        m = compute_selective_metrics_at_threshold(val_y_true, val_y_pred, val_confidences, th)
        if m["accepted_count"] > 0 and m["selective_risk"] <= target_risk_budget:
            if m["coverage"] > best_coverage:
                best_coverage = m["coverage"]
                best_threshold = float(th)
                achieved_val_risk = m["selective_risk"]

    if best_threshold is None:
        # If no threshold satisfies target risk, pick max threshold (safest abstention)
        best_threshold = float(thresholds[-1])
        m_max = compute_selective_metrics_at_threshold(val_y_true, val_y_pred, val_confidences, best_threshold)
        best_coverage = m_max["coverage"]
        achieved_val_risk = m_max["selective_risk"]

    return {
        "target_risk_budget": target_risk_budget,
        "optimal_threshold": best_threshold,
        "val_coverage": best_coverage,
        "val_selective_risk": achieved_val_risk
    }
