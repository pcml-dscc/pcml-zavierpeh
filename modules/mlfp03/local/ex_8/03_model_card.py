# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 8.3: Mitchell et al. Model Cards for Regulated ML
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Structure of the Mitchell et al. Model Card (9 sections)
#   - Auto-generating a model card from training artifacts
#   - Distinguishing "intended use" from "out of scope"
#   - Writing the ethics + limitations sections regulators actually read
#   - Rendering the card as a visual summary for non-technical reviewers
#
# PREREQUISITES: Exercise 8.1 (reuses the calibrated model).
#
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Theory     — why model cards exist (EU AI Act, MAS FEAT, Mitchell)
#   2. Build      — the 9-section template filled from training artifacts
#   3. Train      — (no training) render the card to markdown
#   4. Visualise  — a single-page model-card summary diagram
#   5. Apply      — MAS FEAT compliance for a high-risk credit model
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from shared.mlfp03.ex_8 import (
    OUTPUT_DIR,
    evaluate_classification,
    load_credit_split,
    train_calibrated_model,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Model Cards Exist
# ════════════════════════════════════════════════════════════════════════
# Mitchell et al. (2019) proposed Model Cards after a string of ML
# failures where the model worked as built but was DEPLOYED outside its
# intended context. The card is a one-page contract: what this model
# does, where it's been validated, where it must NOT be used, how to
# tell when it stops working.
#
# Regulatory context: EU AI Act Article 13, Singapore MAS FEAT
# principles, US NIST AI RMF — all require model cards for high-risk AI.
#
# The 9 SECTIONS:
#   1. Model details     2. Intended use      3. Factors
#   4. Metrics           5. Evaluation data   6. Training data
#   7. Quantitative      8. Ethical           9. Caveats


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the model card template from training artifacts
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 8.3 — Model Card Generation")
print("=" * 70)

split = load_credit_split()
X_train, y_train = split["X_train"], split["y_train"]
X_test, y_test = split["X_test"], split["y_test"]

# TODO: Train the calibrated model and score the test set
calibrated_model = ____
y_proba = ____
metrics = evaluate_classification(y_test, y_proba)

# Conformal coverage (short re-derivation so this file runs standalone)
n_cal = X_test.shape[0] // 2
cal_proba = calibrated_model.predict_proba(X_test[:n_cal])[:, 1]
cal_scores = np.where(y_test[:n_cal] == 1, 1 - cal_proba, cal_proba)
alpha = 0.10
q_level = np.ceil((len(cal_scores) + 1) * (1 - alpha)) / len(cal_scores)
q_hat = float(np.quantile(cal_scores, min(q_level, 1.0)))
eval_proba = calibrated_model.predict_proba(X_test[n_cal:])[:, 1]
y_eval = y_test[n_cal:]
correct_sets = [
    (y_eval[i] == 1 and (1 - eval_proba[i]) <= q_hat)
    or (y_eval[i] == 0 and eval_proba[i] <= q_hat)
    for i in range(len(y_eval))
]
coverage = float(np.mean(correct_sets))

print(
    f"\nAUC-ROC={metrics['auc_roc']:.4f}  Brier={metrics['brier']:.4f}  Coverage={coverage:.3f}"
)

# TODO: Fill in the f-string model card below. Every section must be
# present — the checkpoint below scans for the 9 Mitchell section names.
model_card = f"""
# Model Card: Singapore Credit Default Prediction

## 1. Model Details
- **Model type**: LightGBM Classifier + isotonic calibration
- **Version**: 1.0
- **Date**: {datetime.now().strftime("%Y-%m-%d")}
- **Framework**: kailash-ml (Terrene Foundation)
- **License**: Apache 2.0
- **Contact**: model-risk@example-sg.org

## 2. Intended Use
- **Primary use**: Singapore retail unsecured credit default risk
- **Primary users**: Credit analysts; automated underwriting with human escalation
- **Out of scope**: Commercial credit, cross-border lending, regulatory capital

## 3. Factors
- **Groups evaluated**: age, gender, residency status, income bands
- **Instruments**: application forms + bureau data
- **Environments**: online + branch application intake

## 4. Metrics
- **Measures**: AUC-ROC, AUC-PR, Brier, F1
- **Decision thresholds**: conformal α=0.10 prediction set routing
- **Fairness measures**: disparate impact, equalised odds gap

## 5. Evaluation Data
- **Source**: held-out 20% of Singapore credit applications
- **Size**: {X_test.shape[0]:,} samples
- **Preprocessing**: kailash-ml PreprocessingPipeline (ordinal encoding)

## 6. Training Data
- **Source**: Singapore credit applications
- **Size**: {X_train.shape[0]:,} samples / {X_train.shape[1]} features
- **Target**: Binary default ({y_train.mean():.1%} positive)
- **Time range**: 2020-2024

## 7. Quantitative Analyses
- **AUC-ROC**: {metrics['auc_roc']:.4f}
- **AUC-PR**: {metrics['auc_pr']:.4f}
- **Brier**: {metrics['brier']:.4f}
- **F1**: {metrics['f1']:.4f}
- **Conformal coverage**: {coverage:.1%} at α={alpha}

## 8. Ethical Considerations
- Protected attributes analysed with SHAP
- Disparate impact within MAS FEAT 0.8-1.25 band
- Contestability: every adverse decision includes top-5 SHAP reasons
- Ambiguous cases routed to human review

## 9. Caveats and Recommendations
- Retrain when PSI > 0.2 OR AUC-PR < {metrics['auc_pr'] * 0.9:.4f}
- Exchangeability assumption breaks under adversarial drift
- Singapore-only — do not apply to other markets without retraining
- Sunset at version 2.0 or 12 months, whichever first
"""

# TODO: Write model_card to OUTPUT_DIR / "ex8_03_model_card.md"
card_path = OUTPUT_DIR / "ex8_03_model_card.md"
____


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert card_path.exists(), "Task 2: Model card should be written"
required_sections = [
    "Model Details",
    "Intended Use",
    "Factors",
    "Metrics",
    "Evaluation Data",
    "Training Data",
    "Quantitative Analyses",
    "Ethical Considerations",
    "Caveats and Recommendations",
]
for section in required_sections:
    assert section in model_card, f"Task 2: Missing section '{section}'"
print("\n[ok] Checkpoint 1 — all 9 Mitchell sections present\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — render the card (no training)
# ════════════════════════════════════════════════════════════════════════

print("=== Model Card (excerpt) ===")
for line in model_card.splitlines()[:25]:
    print(line)
print("  ... (full card written to disk)")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the card as a one-page summary
# ════════════════════════════════════════════════════════════════════════

fig = go.Figure()
# TODO: Add four Indicator gauges for AUC-ROC, AUC-PR, Brier, Coverage.
# Use grid positions {"row": 0-1, "column": 0-1} to make a 2x2 panel.
fig.add_trace(
    go.Indicator(
        mode="gauge+number",
        value=metrics["auc_roc"],
        title={"text": "AUC-ROC"},
        gauge={"axis": {"range": [0.5, 1.0]}, "bar": {"color": "#2563eb"}},
        domain={"row": 0, "column": 0},
    )
)
fig.add_trace(
    go.Indicator(
        mode="gauge+number",
        value=metrics["auc_pr"],
        title={"text": "AUC-PR"},
        gauge={"axis": {"range": [0.0, 1.0]}, "bar": {"color": "#10b981"}},
        domain={"row": 0, "column": 1},
    )
)
# TODO: Add the Brier indicator (row=1, column=0, range 0-0.25, lower is better)
____
# TODO: Add the Coverage indicator (row=1, column=1, range 0-1)
____
fig.update_layout(
    grid={"rows": 2, "columns": 2, "pattern": "independent"},
    title="Model Card Summary — Singapore Credit Default v1.0",
    height=560,
)
viz_path = OUTPUT_DIR / "ex8_03_model_card_summary.html"
fig.write_html(str(viz_path))
print(f"\nSaved: {viz_path}")


# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert Path(viz_path).exists(), "Task 4: Summary visual should be written"
print("\n[ok] Checkpoint 2 — one-page visual summary rendered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: MAS FEAT compliance savings at UOB scale
# ════════════════════════════════════════════════════════════════════════
# Manual card authoring: 40h/model × 180 models × S$120/h × 4q = ~S$3.5M/yr
# Automated card authoring: 4h/model same math = ~S$350K/yr
# Annual savings: ~S$3.1M + avoided MAS deficiency exposure (up to S$1M/model).

hours_manual = 40
hours_automated = 4
models = 180
hourly = 120.0
annual_savings = (hours_manual - hours_automated) * models * hourly * 4
print(f"\n=== UOB FEAT compliance savings ===")
print(f"  Annual savings: S${annual_savings:,.0f}/yr")
print(f"  Plus: avoided MAS deficiency exposure (up to S$1M per model)")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] 9 Mitchell et al. model card sections
  [x] Auto-generated card from training artifacts
  [x] One-page visual summary for non-technical reviewers
  [x] MAS FEAT compliance: ~S${annual_savings:,.0f}/yr at UOB scale

  Next: 04_deployment_pipeline.py — register, version, promote.
"""
)
