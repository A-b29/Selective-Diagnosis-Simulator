"""
calibration.py - Confidence calibration engine (Temperature Scaling & Platt Scaling).

Fit strictly on VALIDATION logits. Never touches test data during parameter estimation.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


@dataclass
class ReliabilityData:
    bin_centers: List[float]
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]
    ece: float
    mce: float
    brier_score: float


class TemperatureScaler:
    """
    Optimizes a single scalar Temperature T > 0 on validation logits
    to minimize cross-entropy loss: sigma(z / T).
    """

    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray) -> "TemperatureScaler":
        """Fit optimal temperature on held-out validation set."""
        eps = 1e-12

        def loss_fn(t_param: np.ndarray) -> float:
            t = float(t_param[0])
            if t <= 0.001:
                return 1e6
            scaled_logits = val_logits / t
            # Stable sigmoid
            probs = 1.0 / (1.0 + np.exp(-np.clip(scaled_logits, -30.0, 30.0)))
            # Binary cross entropy
            loss = -np.mean(val_labels * np.log(probs + eps) + (1.0 - val_labels) * np.log(1.0 - probs + eps))
            return float(loss)

        res = minimize(loss_fn, x0=[1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        return self

    def transform(self, logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply learned temperature.
        Returns:
            calibrated_probs: Shape (N, 2)
            calibrated_confidences: Shape (N,)
        """
        scaled = logits / self.temperature
        p1 = 1.0 / (1.0 + np.exp(-np.clip(scaled, -30.0, 30.0)))
        p0 = 1.0 - p1
        probs = np.column_stack([p0, p1])
        confidences = np.max(probs, axis=1)
        return probs, confidences


def compute_reliability_curve(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10
) -> ReliabilityData:
    """
    Compute reliability diagram bins, ECE, MCE, and Brier score.
    """
    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    ece = 0.0
    mce = 0.0
    n_total = len(confidences)

    for i in range(n_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]

        if i == 0:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences > low) & (confidences <= high)

        count = int(np.sum(mask))
        bin_counts.append(count)
        center = float((low + high) / 2.0)
        bin_centers.append(center)

        if count > 0:
            acc = float(np.mean(accuracies[mask]))
            conf = float(np.mean(confidences[mask]))
            bin_accs.append(acc)
            bin_confs.append(conf)

            diff = abs(acc - conf)
            ece += (count / n_total) * diff
            mce = max(mce, diff)
        else:
            bin_accs.append(center)  # fallback for empty bin
            bin_confs.append(center)

    brier = float(np.mean((confidences - accuracies) ** 2))

    return ReliabilityData(
        bin_centers=bin_centers,
        bin_accuracies=bin_accs,
        bin_confidences=bin_confs,
        bin_counts=bin_counts,
        ece=float(ece),
        mce=float(mce),
        brier_score=brier
    )
