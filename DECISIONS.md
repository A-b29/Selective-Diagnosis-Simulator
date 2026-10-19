# Architectural & Cunning Decisions: Selective Diagnosis Simulator

## 1. Zero Test-Set Tuning for Abstention Thresholds
**Decision:** All abstention thresholds $\theta^*$ are strictly derived using the held-out **Validation set** given a target risk budget $\epsilon$. The test partition remains locked until evaluation.
**Cunning Rationale:**
- Optimizing an abstention threshold on the test set produces artificially optimistic coverage figures (data snooping / target leakage).
- By calibrating temperature and tuning $\theta^*$ solely on validation data, our test coverage and test selective risk represent genuine, prospective generalization.

## 2. Temperature Scaling Calibration Isolation
**Decision:** We implement parametric post-hoc temperature scaling ($\sigma(z / T)$) fit via L-BFGS-B optimization on validation cross-entropy loss.
**Cunning Rationale:**
- Deep models and standard classifiers tend to produce overconfident probabilities (uncalibrated softmax).
- Using uncalibrated confidences for abstention distorts the coverage-risk curve because raw probabilities do not correspond to empirical correctness rates.
- Calibration aligns confidence with accuracy, lowering Expected Calibration Error (ECE) and yielding predictable error budgets.

## 3. Strict Preprocessing Isolation
**Decision:** `StandardScaler` is fitted strictly on `X_train`. `X_val` and `X_test` are only transformed.
**Cunning Rationale:**
- Fitting feature scalers across the full dataset introduces mean/variance leakage across folds, inflating classification accuracy. Our automated tests (`test_data_leakage.py`) explicitly verify `scaler.n_samples_seen_ == len(y_train)`.

## 4. Handling Undefined Risk in Zero-Accepted Edge Cases
**Decision:** When threshold $\theta$ is set so high that zero cases are accepted ($N_{accepted} = 0$), selective risk is defined mathematically as 0.0 with $N_{accepted} = 0$, avoiding runtime division-by-zero crashes.
