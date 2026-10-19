# Selective Diagnosis Simulator

A simulator and evaluation harness studying the mathematical and empirical trade-off between prediction coverage and selective risk when a biomedical model can abstain and refer uncertain cases to expert review.

> **Research Prototype Notice:** Designed for algorithmic uncertainty and calibration analysis. Not an approved clinical decision support system.

---

## Features
- **Strict Zero-Leakage Protocol:** Feature scalers, temperature scalers, and abstention thresholds are fitted exclusively on training/validation partitions.
- **Parametric Temperature Scaling:** Minimizes validation negative log-likelihood to recalibrate overconfident model logits.
- **Coverage vs. Selective Risk Curve:** Maps out full operational curves across confidence thresholds.
- **Risk Budget Optimizer:** Automatically selects threshold $\theta^*$ on validation data to achieve a user-defined error budget (e.g. $\le 2\%$).
- **Dual Interface:** Streamlit interactive simulator and headless CLI evaluation runner.

---

## Quickstart

### 1. Run Unit Tests
```bash
pytest tests/ -v
```

### 2. Run CLI Evaluation
```bash
python cli.py --risk-budget 0.02 --output-dir reports/
```

### 3. Launch Streamlit Web Simulator
```bash
streamlit run app.py
```

---

## File Structure
```text
selective_diagnosis_simulator/
├── app.py                      # Streamlit interactive simulator
├── cli.py                      # Headless evaluation CLI
├── DECISIONS.md                # Cunning and architectural decisions
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── src/
│   ├── data.py                 # Biomedical dataset with strict train/val/test splits
│   ├── train.py                # Classifier training & logit extraction
│   ├── calibration.py          # Temperature scaling, ECE, and reliability diagrams
│   ├── selective_metrics.py    # Coverage vs selective risk curve & threshold optimizer
│   └── reporting.py            # Markdown and JSON report generator
└── tests/
    ├── conftest.py
    ├── test_data_leakage.py    # Zero leakage unit tests
    ├── test_calibration.py     # Calibration & ECE unit tests
    └── test_selective_metrics.py # Abstention & edge case tests
```
