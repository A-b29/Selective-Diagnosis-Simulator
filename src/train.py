"""
train.py - Baseline classifier training and raw logit extraction.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


@dataclass
class ModelPredictions:
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray  # Shape: (N, 2)
    confidences: np.ndarray    # Max class probability: Shape: (N,)
    raw_logits: np.ndarray     # Shape: (N,)


def train_baseline_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "logistic_regression",
    seed: int = 42
) -> Any:
    """Train a supervised biomedical baseline classifier."""
    if model_type == "logistic_regression":
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)
    return model


def get_predictions(model: Any, X: np.ndarray, y: np.ndarray) -> ModelPredictions:
    """Generate predictions, class probabilities, and decision margin logits."""
    probs = model.predict_proba(X)
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    # Extract log-odds / logits: log(p1 / p0)
    eps = 1e-7
    p1 = np.clip(probs[:, 1], eps, 1.0 - eps)
    logits = np.log(p1 / (1.0 - p1))

    return ModelPredictions(
        y_true=y,
        y_pred=preds,
        probabilities=probs,
        confidences=confidences,
        raw_logits=logits
    )


def evaluate_baseline(preds: ModelPredictions) -> Dict[str, float]:
    """Calculate standard classification evaluation metrics."""
    acc = float(accuracy_score(preds.y_true, preds.y_pred))
    f1 = float(f1_score(preds.y_true, preds.y_pred, average="weighted"))
    auc = float(roc_auc_score(preds.y_true, preds.probabilities[:, 1]))
    return {
        "accuracy": acc,
        "f1_weighted": f1,
        "auroc": auc,
        "error_rate": float(1.0 - acc)
    }
