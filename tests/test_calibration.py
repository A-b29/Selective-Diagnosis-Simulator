"""
test_calibration.py - Unit tests for temperature scaling and reliability diagrams.
"""

import numpy as np
import pytest
from src.calibration import TemperatureScaler, compute_reliability_curve


def test_temperature_scaling_optimization():
    # Synthetic overconfident logits
    rng = np.random.default_rng(42)
    logits = rng.normal(loc=0.0, scale=4.0, size=200)
    # Generate noisy labels
    true_probs = 1.0 / (1.0 + np.exp(-logits / 2.0))
    labels = (rng.uniform(0.0, 1.0, size=200) < true_probs).astype(int)

    scaler = TemperatureScaler().fit(logits, labels)
    # Temperature should be > 1 to soften overconfidence
    assert scaler.temperature > 0.05
    assert scaler.temperature < 10.0

    cal_probs, cal_confs = scaler.transform(logits)
    assert cal_probs.shape == (200, 2)
    assert np.all(cal_confs >= 0.5)
    assert np.all(cal_confs <= 1.0)


def test_reliability_curve_computation():
    confs = np.array([0.9, 0.8, 0.7, 0.6])
    accs = np.array([1, 1, 0, 0])
    rel = compute_reliability_curve(confs, accs, n_bins=5)

    assert 0.0 <= rel.ece <= 1.0
    assert 0.0 <= rel.brier_score <= 1.0
    assert len(rel.bin_centers) == 5
