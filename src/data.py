"""
data.py - Biomedical dataset loader with strict train/validation/test isolation.
Ensures zero data leakage: transformers are fit strictly on training partitions.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    target_names: List[str]
    scaler: StandardScaler


def load_biomedical_dataset(
    seed: int = 42,
    val_size: float = 0.20,
    test_size: float = 0.20
) -> DatasetSplits:
    """
    Load Breast Cancer Wisconsin Diagnostic benchmark.
    Features: 30 cell nucleus morphometric descriptors.
    Targets: 0 = Malignant, 1 = Benign.
    
    Split protocol:
    1. Split off test_size (e.g. 20%) stratified by target.
    2. Split remaining 80% into train and validation (val_size / (1 - test_size)).
    3. Fit StandardScaler exclusively on train data; apply to val and test.
    """
    raw = load_breast_cancer()
    X = raw.data.astype(np.float32)
    y = raw.target.astype(np.int64)
    feature_names = list(raw.feature_names)
    target_names = ["Malignant", "Benign"]

    # Step 1: Stratified split for test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Step 2: Stratified split for validation set
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, random_state=seed, stratify=y_temp
    )

    # Step 3: Strict feature scaling without data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return DatasetSplits(
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_val_scaled,
        y_val=y_val,
        X_test=X_test_scaled,
        y_test=y_test,
        feature_names=feature_names,
        target_names=target_names,
        scaler=scaler
    )
