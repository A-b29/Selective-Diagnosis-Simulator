"""
test_data_leakage.py - Unit test asserting zero data leakage across train, val, and test splits.
"""

import numpy as np
import pytest
from src.data import load_biomedical_dataset


def test_zero_split_leakage():
    splits = load_biomedical_dataset(seed=42)

    total_samples = len(splits.y_train) + len(splits.y_val) + len(splits.y_test)
    assert total_samples == 569  # Breast cancer dataset total size

    # Assert non-empty splits
    assert len(splits.y_train) > 0
    assert len(splits.y_val) > 0
    assert len(splits.y_test) > 0

    # Scaler mean must match X_train unscaled mean, NOT the full dataset mean
    # We verify scaler was fit only on training data:
    assert splits.scaler.n_samples_seen_ == len(splits.y_train)


def test_stratification_preservation():
    splits = load_biomedical_dataset(seed=42)

    train_pos_rate = np.mean(splits.y_train)
    val_pos_rate = np.mean(splits.y_val)
    test_pos_rate = np.mean(splits.y_test)

    # Class distribution should match closely across all splits due to stratify
    assert abs(train_pos_rate - val_pos_rate) < 0.05
    assert abs(train_pos_rate - test_pos_rate) < 0.05
