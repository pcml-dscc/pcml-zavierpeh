# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP03 Exercise 5 — Class Imbalance & Calibration.

Contains: Singapore credit scoring data loader, Kailash PreprocessingPipeline
wiring, cost-matrix constants, metric helpers, reliability-diagram helpers,
and an OUTPUT_DIR convention for every technique file to write visual proof
to the same place.

Technique-specific code (SMOTE call, focal loss gradient, Platt/Isotonic
model wiring) does NOT belong here — it lives in the per-technique files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from kailash_ml import PreprocessingPipeline
from kailash_ml.interop import to_sklearn_input

from shared.data_loader import MLFPDataLoader

# ════════════════════════════════════════════════════════════════════════
# OUTPUT DIRECTORY — every technique writes visual proof to the same place
# ════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("outputs") / "ex5_imbalance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# BUSINESS CONTEXT — Singapore retail bank credit scoring
# ════════════════════════════════════════════════════════════════════════
# These constants drive every technique file. A 100:1 cost ratio is
# realistic for SEA consumer lending: the average charged-off unsecured
# loan in Singapore is ~S$10,000 (MAS consumer credit report 2024), and
# the operational cost of a false decline (manual review + lost NPV of
# the customer relationship) is roughly S$100.


@dataclass(frozen=True)
class CostMatrix:
    """Dollar cost of each confusion-matrix cell.

    fn = cost of missing a default (charge-off loss)
    fp = cost of a false alarm (manual review + lost relationship NPV)
    """

    fn: float = 10_000.0
    fp: float = 100.0

    @property
    def optimal_threshold(self) -> float:
        """Bayes-optimal threshold for this cost matrix: t* = fp / (fp + fn)."""
        return self.fp / (self.fp + self.fn)

    def total_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return float(fp * self.fp + fn * self.fn)


DEFAULT_COSTS = CostMatrix(fn=10_000.0, fp=100.0)

# Annual volume for ROI analysis — calibrated to a mid-tier SG retail bank.
ANNUAL_APPLICATIONS = 100_000


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — Singapore credit scoring dataset
# ════════════════════════════════════════════════════════════════════════
# The dataset is loaded through the MLFPDataLoader so it works identically
# in local (.data_cache) and Colab (Drive + gdown) formats.


def load_credit_splits(
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Load the SG credit scoring dataset and return (X_train, y_train, X_test, y_test, pos_rate).

    Uses kailash-ml PreprocessingPipeline for consistent preprocessing across
    every technique file. Returns numpy arrays ready for sklearn-style fit.
    """
    loader = MLFPDataLoader()
    credit = loader.load("mlfp02", "sg_credit_scoring.parquet")

    pipeline = PreprocessingPipeline()
    result = pipeline.setup(
        credit,
        target="default",
        seed=seed,
        normalize=False,
        categorical_encoding="ordinal",
    )

    feature_cols = [c for c in result.train_data.columns if c != "default"]
    X_train, y_train, _ = to_sklearn_input(
        result.train_data,
        feature_columns=feature_cols,
        target_column="default",
    )
    X_test, y_test, _ = to_sklearn_input(
        result.test_data,
        feature_columns=feature_cols,
        target_column="default",
    )
    pos_rate = float(y_train.mean())
    return X_train, y_train, X_test, y_test, pos_rate


# ════════════════════════════════════════════════════════════════════════
# METRIC HELPERS — complete taxonomy in one call
# ════════════════════════════════════════════════════════════════════════


def metrics_row(
    name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute a full metrics row for a given strategy name + probabilities.

    Returns a dict compatible with polars DataFrame construction so every
    technique file can build comparison tables with identical column shapes.
    """
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "strategy": name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "auc_pr": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def print_metrics_table(rows: list[dict[str, Any]], title: str) -> None:
    """Print a comparison table across strategies."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(
        f"{'Strategy':<22} {'AUC-PR':>8} {'Brier':>8} {'F1':>8} "
        f"{'Precision':>10} {'Recall':>8}"
    )
    print("─" * 70)
    for r in rows:
        print(
            f"{r['strategy']:<22} {r['auc_pr']:>8.4f} {r['brier']:>8.4f} "
            f"{r['f1']:>8.4f} {r['precision']:>10.4f} {r['recall']:>8.4f}"
        )


def rows_to_dataframe(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Convert metrics rows to a polars DataFrame (for persistence / plots)."""
    return pl.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════
# RELIABILITY DIAGRAM DATA (calibration curve, binned)
# ════════════════════════════════════════════════════════════════════════


def reliability_bins(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> pl.DataFrame:
    """Compute a binned reliability diagram as a polars DataFrame.

    Columns: bin_lower, bin_upper, mean_pred, empirical_rate, count, gap
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_proba >= lo) & (y_proba < hi if i < n_bins - 1 else y_proba <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(y_proba[mask].mean())
        empirical = float(y_true[mask].mean())
        rows.append(
            {
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "mean_pred": mean_pred,
                "empirical_rate": empirical,
                "count": count,
                "gap": float(abs(mean_pred - empirical)),
            }
        )
    return pl.DataFrame(rows)


def print_reliability(name: str, bins: pl.DataFrame) -> None:
    print(f"\n  Reliability bins — {name}")
    print(f"  {'mean_pred':>10} {'empirical':>10} {'|gap|':>8} {'n':>6}")
    print("  " + "─" * 38)
    for row in bins.iter_rows(named=True):
        print(
            f"  {row['mean_pred']:>10.3f} {row['empirical_rate']:>10.3f} "
            f"{row['gap']:>8.3f} {row['count']:>6}"
        )


# ════════════════════════════════════════════════════════════════════════
# BUSINESS ROI CALCULATOR
# ════════════════════════════════════════════════════════════════════════


def annual_roi(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    costs: CostMatrix = DEFAULT_COSTS,
    annual_volume: int = ANNUAL_APPLICATIONS,
) -> dict[str, float]:
    """Project test-set confusion matrix onto an annual volume.

    Returns a dict with caught_defaults, missed_defaults, false_alarms,
    model_cost, no_model_cost, annual_savings — all in dollars.
    """
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    pos_rate = float(y_true.mean())
    scale = annual_volume / len(y_true)
    annual_fn = fn * scale
    annual_fp = fp * scale
    annual_tp = tp * scale
    n_defaults_annual = pos_rate * annual_volume
    model_cost = annual_fn * costs.fn + annual_fp * costs.fp
    no_model_cost = n_defaults_annual * costs.fn
    return {
        "threshold": float(threshold),
        "defaults_caught": float(annual_tp),
        "defaults_missed": float(annual_fn),
        "false_alarms": float(annual_fp),
        "model_cost_usd": float(model_cost),
        "no_model_cost_usd": float(no_model_cost),
        "annual_savings_usd": float(no_model_cost - model_cost),
        "annual_volume": int(annual_volume),
    }


def print_roi(name: str, roi: dict[str, float]) -> None:
    print(f"\n  ROI — {name}")
    print(f"    Threshold:          {roi['threshold']:.4f}")
    print(f"    Defaults caught:    {roi['defaults_caught']:>12,.0f}")
    print(f"    Defaults missed:    {roi['defaults_missed']:>12,.0f}")
    print(f"    False alarms:       {roi['false_alarms']:>12,.0f}")
    print(f"    Model cost:         ${roi['model_cost_usd']:>12,.0f}")
    print(f"    No-model cost:      ${roi['no_model_cost_usd']:>12,.0f}")
    print(f"    Annual savings:     ${roi['annual_savings_usd']:>12,.0f}")


# ════════════════════════════════════════════════════════════════════════
# PERSISTENCE — shared parquet of per-strategy probabilities
# ════════════════════════════════════════════════════════════════════════
# Each technique file writes its y_proba vector to a parquet under
# OUTPUT_DIR so that later technique files (threshold opt, calibration,
# final comparison) can read them without re-training.

PROBA_STORE = OUTPUT_DIR / "strategy_probabilities.parquet"


def save_strategy_proba(name: str, y_proba: np.ndarray) -> None:
    """Upsert a strategy's probability vector into the shared parquet store."""
    new_df = pl.DataFrame({"strategy": [name] * len(y_proba), "y_proba": y_proba})
    if PROBA_STORE.exists():
        existing = pl.read_parquet(PROBA_STORE)
        existing = existing.filter(pl.col("strategy") != name)
        combined = pl.concat([existing, new_df])
    else:
        combined = new_df
    combined.write_parquet(PROBA_STORE)


def load_strategy_proba(name: str) -> np.ndarray:
    """Read back a previously-saved probability vector for a strategy."""
    if not PROBA_STORE.exists():
        raise FileNotFoundError(
            f"{PROBA_STORE} not found — run the earlier technique files first."
        )
    df = pl.read_parquet(PROBA_STORE).filter(pl.col("strategy") == name)
    if df.height == 0:
        raise KeyError(f"Strategy {name!r} not found in {PROBA_STORE}")
    return df["y_proba"].to_numpy()


def list_saved_strategies() -> list[str]:
    if not PROBA_STORE.exists():
        return []
    df = pl.read_parquet(PROBA_STORE)
    return df["strategy"].unique().to_list()
