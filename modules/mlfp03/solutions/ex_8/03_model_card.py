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
# PREREQUISITES: Exercise 8.1 (reuses the calibrated model). Fairness
# audit from Module 3 Exercise 6 is referenced but not re-run here.
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
# Mitchell et al. (2019) proposed Model Cards after a string of high-
# profile ML failures where the model worked as built but was DEPLOYED
# outside its intended context (facial recognition trained on lighter-
# skinned faces, pneumonia models relying on hospital-ID leakage, etc.).
#
# The model card is a one-page contract between the ML team and everyone
# downstream: what this model does, where it's been validated, where it
# MUST NOT be used, and how to tell when it stops working.
#
# Regulatory context (2024-2026):
#   - EU AI Act Article 13: transparency + documentation for high-risk AI
#   - Singapore MAS FEAT principles (Fairness, Ethics, Accountability,
#     Transparency) require model cards for consumer credit decisioning
#   - US NIST AI RMF: model cards are a recommended artifact for the
#     GOVERN and MANAGE functions
#
# The 9 SECTIONS (Mitchell et al. 2019, Table 1):
#   1. Model details        — type, version, date, contact
#   2. Intended use         — primary users, primary purpose, scope
#   3. Factors              — groups, instruments, environments
#   4. Metrics              — evaluation measures + decision thresholds
#   5. Evaluation data      — source, preprocessing, motivation
#   6. Training data        — same categories as evaluation
#   7. Quantitative analyses— disaggregated + intersectional results
#   8. Ethical considerations— risks, mitigations, dual-use concerns
#   9. Caveats & recommendations — out-of-scope uses, future work


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the model card template from training artifacts
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP03 Exercise 8.3 — Model Card Generation")
print("=" * 70)

split = load_credit_split()
X_train, y_train = split["X_train"], split["y_train"]
X_test, y_test = split["X_test"], split["y_test"]

calibrated_model = train_calibrated_model(X_train, y_train)
y_proba = calibrated_model.predict_proba(X_test)[:, 1]
metrics = evaluate_classification(y_test, y_proba)

# Conformal coverage (short re-derivation so this file runs standalone)
import numpy as np

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
    f"\nAUC-ROC={metrics['auc_roc']:.4f}  Brier={metrics['brier']:.4f}  "
    f"Coverage={coverage:.3f}"
)


model_card = f"""
# Model Card: Singapore Credit Default Prediction

## 1. Model Details
- **Model type**: LightGBM Classifier (500 trees, depth 7) + isotonic calibration
- **Version**: 1.0
- **Date**: {datetime.now().strftime("%Y-%m-%d")}
- **Framework**: kailash-ml (Terrene Foundation)
- **License**: Apache 2.0
- **Contact**: model-risk@example-sg.org

## 2. Intended Use
- **Primary use**: Unsecured consumer credit default risk assessment for
  the Singapore retail banking market.
- **Primary users**: Credit risk analysts; automated underwriting systems
  that escalate ambiguous applications to humans.
- **Out of scope**:
  - Corporate/commercial credit decisions
  - Cross-border lending (model is trained on SG residents only)
  - Regulatory capital calculation (use dedicated PD/LGD/EAD models)
  - Any decision with irreversible consequences without human review

## 3. Factors
- **Groups evaluated**: age buckets, gender, residency status, income
  bands (see Exercise 6 fairness audit)
- **Instruments**: DBS/OCBC/UOB application forms, bureau pull data
- **Environments**: production application intake (online + branch)

## 4. Metrics
- **Evaluation measures**: AUC-ROC, AUC-PR, Brier score, F1
- **Decision thresholds**: conformal prediction set routing at α=0.10
  (auto-decide when singleton, route to human when ambiguous)
- **Fairness measures**: disparate impact ratio, equalised odds gap

## 5. Evaluation Data
- **Source**: held-out 20% slice of Singapore credit applications
- **Size**: {X_test.shape[0]:,} samples
- **Preprocessing**: kailash-ml PreprocessingPipeline with ordinal encoding
- **Motivation**: time-ordered split to approximate deployment conditions

## 6. Training Data
- **Source**: Singapore credit applications (data.gov.sg characteristics)
- **Size**: {X_train.shape[0]:,} samples
- **Features**: {X_train.shape[1]} features (financial, behavioural, demographic)
- **Target**: Binary default ({y_train.mean():.1%} positive rate)
- **Time range**: 2020-2024

## 7. Quantitative Analyses
### Aggregate
- **AUC-ROC**: {metrics['auc_roc']:.4f}
- **AUC-PR**: {metrics['auc_pr']:.4f}
- **Brier Score**: {metrics['brier']:.4f} (calibrated)
- **F1**: {metrics['f1']:.4f}
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}

### Uncertainty Quantification
- **Method**: Split conformal prediction (distribution-free)
- **Coverage**: {coverage:.1%} at α={alpha}
- **Guarantee**: P(Y ∈ C(X)) ≥ 1-α, requires exchangeability

### Disaggregated (see Exercise 6 report)
- Fairness audit: disparate impact within 0.8-1.25 band
- Impossibility theorem acknowledged: cannot satisfy all fairness criteria
- Protected-attribute SHAP analysis in `ex6_fairness.html`

## 8. Ethical Considerations
- **Protected attributes**: age, gender, ethnicity analysed with SHAP
- **Disparate impact**: tested and within MAS FEAT recommended band
- **Dual use**: model outputs MUST NOT be used for marketing segmentation
- **Contestability**: every adverse decision includes a top-5 SHAP reason
  list and a human-appeal pathway (MAS Notice 635 §24)
- **Ambiguous cases**: conformal prediction sets of size 2 route to human
  review — the model refuses to commit on borderline cases

## 9. Caveats and Recommendations
- **Data freshness**: retrain when PSI on any feature > 0.2 OR live
  AUC-PR drops below {metrics['auc_pr'] * 0.9:.4f} (10% degradation floor)
- **Exchangeability**: coverage guarantee breaks under adversarial drift
  (fraud rings); pair with DriftMonitor (Exercise 8.2)
- **Market specificity**: trained on SG retail; DO NOT apply elsewhere
- **Monitoring cadence**: daily DriftMonitor + weekly fairness re-audit
- **Sunset criterion**: model retires at version 2.0 or 12 months,
  whichever comes first
"""

card_path = OUTPUT_DIR / "ex8_03_model_card.md"
card_path.write_text(model_card)
print(f"\nSaved: {card_path}")


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
assert "AUC-ROC" in model_card
assert "Coverage" in model_card
print("\n[ok] Checkpoint 1 — all 9 Mitchell sections present\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — render the card (no training)
# ════════════════════════════════════════════════════════════════════════
# A card is only useful if someone READS it. For regulated deployments
# the card must be version-controlled next to the model artifact.

print("=== Model Card (excerpt) ===")
for line in model_card.splitlines()[:25]:
    print(line)
print("  ... (full card written to disk)")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the card as a one-page summary
# ════════════════════════════════════════════════════════════════════════
# Regulators skim. A single-page visual summary of the card — metric
# gauges + a compliance traffic light — gets read where a 3-page markdown
# file does not.

fig = go.Figure()
# Use indicators as a compact metric panel
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
fig.add_trace(
    go.Indicator(
        mode="gauge+number",
        value=metrics["brier"],
        title={"text": "Brier (lower=better)"},
        gauge={"axis": {"range": [0.0, 0.25]}, "bar": {"color": "#f59e0b"}},
        domain={"row": 1, "column": 0},
    )
)
fig.add_trace(
    go.Indicator(
        mode="gauge+number",
        value=coverage,
        title={"text": f"Conformal coverage (target {1 - alpha:.0%})"},
        gauge={"axis": {"range": [0.0, 1.0]}, "bar": {"color": "#8b5cf6"}},
        domain={"row": 1, "column": 1},
    )
)
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
# TASK 5 — APPLY: MAS FEAT compliance for high-risk credit models
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: UOB's model governance team must submit a FEAT principles
# self-assessment for every consumer credit model in production. Before
# model cards, this took 40 analyst-hours per model per quarter (~180
# models × 40h × S$120/h = S$864,000/quarter, plus internal audit time).
#
# With automated model cards generated from training artifacts:
#   - Card generated in minutes during the training pipeline
#   - Quarterly review: verify the card still matches production metrics
#   - Analyst time drops to ~4 hours/model/quarter
#
# SAVINGS: (40 - 4) × 180 × S$120 × 4 quarters = S$3.1M/year at UOB scale.
# Additional avoided risk: regulatory penalty exposure for missing or
# out-of-date documentation (MAS can fine up to S$1M per deficiency).

hours_manual = 40
hours_automated = 4
models = 180
hourly = 120.0
quarterly_savings = (hours_manual - hours_automated) * models * hourly
annual_savings = quarterly_savings * 4
print(f"\n=== UOB FEAT compliance savings ===")
print(
    f"  Manual card authoring:   {hours_manual}h/model × {models} models × 4q = S${hours_manual * models * hourly * 4:,.0f}/yr"
)
print(
    f"  Automated card authoring: {hours_automated}h/model × {models} models × 4q = S${hours_automated * models * hourly * 4:,.0f}/yr"
)
print(f"  Annual savings:           S${annual_savings:,.0f}/yr")
print(f"  Plus: avoided MAS deficiency exposure (up to S$1M per model)")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] The 9 Mitchell et al. model card sections and when to use each
  [x] Auto-generated a card from training artifacts (no copy-paste)
  [x] Rendered a one-page visual summary for non-technical reviewers
  [x] Translated the card into MAS FEAT compliance savings
      (~S${annual_savings:,.0f}/yr at UOB scale)

  KEY INSIGHT: A model card is not paperwork. It's the contract that
  says "if you deploy this model outside these bounds, the ML team is
  not liable and the model is not validated." Without it, every
  deployment is unbounded.

  Next: 04_deployment_pipeline.py — register, version, and promote
  the model with a full audit trail.
"""
)
