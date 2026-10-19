"""
cli.py - Headless training, calibration, and selective evaluation CLI.
"""

import argparse
from pathlib import Path

from src.data import load_biomedical_dataset
from src.train import train_baseline_model, get_predictions, evaluate_baseline
from src.calibration import TemperatureScaler, compute_reliability_curve
from src.selective_metrics import (
    compute_selective_metrics_at_threshold,
    sweep_coverage_risk_curve,
    select_threshold_for_risk_budget
)
from src.reporting import generate_selective_markdown, export_selective_results_json


def run_experiment(output_dir: Path, target_risk: float = 0.03, seed: int = 42):
    print("=== Running Selective Diagnosis Simulation ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Zero-leakage data loading
    splits = load_biomedical_dataset(seed=seed)
    print(f"Data split: Train={len(splits.y_train)}, Val={len(splits.y_val)}, Test={len(splits.y_test)}")

    # 2. Train baseline model
    model = train_baseline_model(splits.X_train, splits.y_train, seed=seed)
    train_preds = get_predictions(model, splits.X_train, splits.y_train)
    val_preds = get_predictions(model, splits.X_val, splits.y_val)
    test_preds = get_predictions(model, splits.X_test, splits.y_test)

    test_baseline = evaluate_baseline(test_preds)
    print(f"Test Baseline Accuracy: {test_baseline['accuracy']:.2%}, Error Rate: {test_baseline['error_rate']:.2%}")

    # 3. Fit Temperature Scaler strictly on VALIDATION logits
    scaler = TemperatureScaler().fit(val_preds.raw_logits, val_preds.y_true)
    print(f"Fitted Temperature (on Val): {scaler.temperature:.3f}")

    # Transform Val and Test
    val_cal_probs, val_cal_confs = scaler.transform(val_preds.raw_logits)
    test_cal_probs, test_cal_confs = scaler.transform(test_preds.raw_logits)

    # Calibration metrics on test set
    test_uncal_rel = compute_reliability_curve(test_preds.confidences, (test_preds.y_pred == test_preds.y_true).astype(int))
    test_cal_rel = compute_reliability_curve(test_cal_confs, (test_preds.y_pred == test_preds.y_true).astype(int))
    print(f"Test ECE: Uncalibrated={test_uncal_rel.ece:.4f} -> Calibrated={test_cal_rel.ece:.4f}")

    # 4. Select threshold on VALIDATION data to meet target risk budget
    val_opt = select_threshold_for_risk_budget(
        splits.y_val, val_preds.y_pred, val_cal_confs, target_risk_budget=target_risk
    )
    th_star = val_opt["optimal_threshold"]
    print(f"Selected Threshold (from Val for risk <= {target_risk:.1%}): theta* = {th_star:.3f}")

    # 5. Evaluate threshold on held-out TEST set
    test_sel = compute_selective_metrics_at_threshold(splits.y_test, test_preds.y_pred, test_cal_confs, th_star)
    print(f"Held-Out Test Results @ theta*={th_star:.3f}:")
    print(f"  Coverage: {test_sel['coverage']:.2%} ({test_sel['accepted_count']} accepted, {test_sel['abstained_count']} abstained)")
    print(f"  Selective Risk: {test_sel['selective_risk']:.2%} ({test_sel['accepted_errors']} errors)")

    # 6. Export Reports
    cal_info = {
        "temperature": scaler.temperature,
        "uncalibrated_ece": test_uncal_rel.ece,
        "calibrated_ece": test_cal_rel.ece,
        "calibrated_brier": test_cal_rel.brier_score,
    }
    sel_info = {
        "target_risk_budget": target_risk,
        "optimal_threshold": th_star,
        "test_coverage": test_sel["coverage"],
        "test_selective_risk": test_sel["selective_risk"],
        "accepted_count": test_sel["accepted_count"],
        "abstained_count": test_sel["abstained_count"],
        "accepted_errors": test_sel["accepted_errors"],
    }
    md_file = generate_selective_markdown(test_baseline, cal_info, sel_info, output_dir / "selective_report.md")
    json_file = export_selective_results_json({
        "baseline": test_baseline,
        "calibration": cal_info,
        "selective": sel_info,
    }, output_dir / "selective_findings.json")

    print(f"\nArtifacts exported:")
    print(f"-> {md_file}")
    print(f"-> {json_file}")


def main():
    parser = argparse.ArgumentParser(description="Selective Diagnosis Simulator CLI")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory for reports.")
    parser.add_argument("--risk-budget", type=float, default=0.03, help="Target risk budget (e.g. 0.03 = max 3%% error).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    run_experiment(Path(args.output_dir), target_risk=args.risk_budget, seed=args.seed)


if __name__ == "__main__":
    main()
