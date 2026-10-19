"""
app.py - Streamlit Interactive Simulator for Selective Diagnosis and Abstention.
"""

from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.data import load_biomedical_dataset
from src.train import train_baseline_model, get_predictions, evaluate_baseline
from src.calibration import TemperatureScaler, compute_reliability_curve
from src.selective_metrics import (
    compute_selective_metrics_at_threshold,
    sweep_coverage_risk_curve,
    select_threshold_for_risk_budget
)
from src.reporting import generate_selective_markdown, export_selective_results_json

st.set_page_config(
    page_title="Selective Diagnosis Simulator",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Selective Diagnosis & Abstention Simulator")
st.markdown("Explore coverage vs. error rate trade-offs when biomedical models can abstain on uncertain cases.")
st.caption("⚠️ Non-Clinical Prototype: Evaluates algorithmic selective risk mechanics, not clinical diagnosis.")

@st.cache_resource
def prepare_experiment(seed: int = 42):
    splits = load_biomedical_dataset(seed=seed)
    model = train_baseline_model(splits.X_train, splits.y_train, seed=seed)

    train_preds = get_predictions(model, splits.X_train, splits.y_train)
    val_preds = get_predictions(model, splits.X_val, splits.y_val)
    test_preds = get_predictions(model, splits.X_test, splits.y_test)

    # Temperature Scaler fit on validation set
    scaler = TemperatureScaler().fit(val_preds.raw_logits, val_preds.y_true)
    val_cal_probs, val_cal_confs = scaler.transform(val_preds.raw_logits)
    test_cal_probs, test_cal_confs = scaler.transform(test_preds.raw_logits)

    return splits, model, train_preds, val_preds, test_preds, scaler, val_cal_confs, test_cal_confs

splits, model, train_preds, val_preds, test_preds, scaler, val_cal_confs, test_cal_confs = prepare_experiment()

# Sidebar: Risk Budget and Abstention Protocol
st.sidebar.header("Abstention Policy")
target_risk = st.sidebar.slider(
    "Target Selective Risk Budget (Max Error %):",
    min_value=0.5,
    max_value=8.0,
    value=2.0,
    step=0.5
) / 100.0

use_calibration = st.sidebar.checkbox("Use Temperature Calibrated Confidences", value=True)

active_val_confs = val_cal_confs if use_calibration else val_preds.confidences
active_test_confs = test_cal_confs if use_calibration else test_preds.confidences

# Compute optimal threshold on VALIDATION set
val_opt = select_threshold_for_risk_budget(
    splits.y_val, val_preds.y_pred, active_val_confs, target_risk_budget=target_risk
)
th_star = val_opt["optimal_threshold"]

# Evaluate on TEST set
test_baseline = evaluate_baseline(test_preds)
test_sel = compute_selective_metrics_at_threshold(splits.y_test, test_preds.y_pred, active_test_confs, th_star)

# High Level Metrics Display
col1, col2, col3, col4 = st.columns(4)
col1.metric("Selected Threshold (θ*)", f"{th_star:.3f}", help="Selected on validation set to satisfy risk budget")
col2.metric("Test Coverage", f"{test_sel['coverage']:.1%}", f"{test_sel['accepted_count']}/{len(splits.y_test)} Accepted")
col3.metric("Selective Risk (Accepted Error)", f"{test_sel['selective_risk']:.1%}", f"Budget: <={target_risk:.1%}")
col4.metric("Baseline Error (100% Coverage)", f"{test_baseline['error_rate']:.1%}")

st.divider()

# Coverage vs Risk Curve
col_left, col_right = st.columns([1.5, 1.2])

with col_left:
    st.subheader("📈 Coverage vs. Selective Risk Trade-Off Curve")
    test_curve = sweep_coverage_risk_curve(splits.y_test, test_preds.y_pred, active_test_confs, num_thresholds=60)
    
    covs = [p["coverage"] * 100 for p in test_curve]
    risks = [p["selective_risk"] * 100 for p in test_curve]

    fig_rc = go.Figure()
    fig_rc.add_trace(go.Scatter(
        x=covs,
        y=risks,
        mode="lines+markers",
        line=dict(color="#2563eb", width=3),
        name="Test Trade-Off"
    ))

    # Mark current operational operating point
    fig_rc.add_trace(go.Scatter(
        x=[test_sel["coverage"] * 100],
        y=[test_sel["selective_risk"] * 100],
        mode="markers",
        marker=dict(size=14, color="#dc2626", symbol="star"),
        name="Operating Point (θ*)"
    ))

    # Risk budget horizontal line
    fig_rc.add_hline(
        y=target_risk * 100,
        line_dash="dash",
        line_color="#16a34a",
        annotation_text=f"Target Risk Budget ({target_risk:.1%})"
    )

    fig_rc.update_layout(
        xaxis_title="Coverage (%) — Accepted Cases",
        yaxis_title="Selective Risk (%) — Error on Accepted",
        height=380,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_rc, use_container_width=True)

with col_right:
    st.subheader("🎯 Calibration Reliability Diagram")
    
    test_correct = (test_preds.y_pred == splits.y_test).astype(int)
    uncal_rel = compute_reliability_curve(test_preds.confidences, test_correct)
    cal_rel = compute_reliability_curve(test_cal_confs, test_correct)

    fig_rel = go.Figure()
    # Ideal diagonal
    fig_rel.add_trace(go.Scatter(
        x=[0.5, 1.0], y=[0.5, 1.0],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Perfect Calibration"
    ))
    fig_rel.add_trace(go.Scatter(
        x=uncal_rel.bin_centers,
        y=uncal_rel.bin_accuracies,
        mode="lines+markers",
        name=f"Uncalibrated (ECE={uncal_rel.ece:.3f})",
        line=dict(color="#f59e0b")
    ))
    fig_rel.add_trace(go.Scatter(
        x=cal_rel.bin_centers,
        y=cal_rel.bin_accuracies,
        mode="lines+markers",
        name=f"Calibrated (ECE={cal_rel.ece:.3f})",
        line=dict(color="#16a34a")
    ))
    fig_rel.update_layout(
        xaxis_title="Confidence Bin Center",
        yaxis_title="Empirical Accuracy",
        height=380,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_rel, use_container_width=True)

# Case Inspection Table
st.divider()
st.subheader("🔎 Case Inspection (Accepted vs. Abstained)")

test_df = pd.DataFrame({
    "Sample Index": np.arange(len(splits.y_test)),
    "True Diagnosis": [splits.target_names[i] for i in splits.y_test],
    "Model Prediction": [splits.target_names[i] for i in test_preds.y_pred],
    "Confidence": [f"{c:.2%}" for c in active_test_confs],
    "Action": ["✅ Accept" if c >= th_star else "🚩 Abstain (Triage)" for c in active_test_confs],
    "Correct": ["Yes" if yt == yp else "No" for yt, yp in zip(splits.y_test, test_preds.y_pred)]
})

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.write("Abstained (Referred for Expert Clinical Review):")
    st.dataframe(test_df[test_df["Action"] == "🚩 Abstain (Triage)"], use_container_width=True)

with col_t2:
    st.write("Accepted Predictions:")
    st.dataframe(test_df[test_df["Action"] == "✅ Accept"], use_container_width=True)
