# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 2: Hypothesis Testing, Bootstrap & CUPED

Complete the `solve()` function. Read problem.md for the full specification.
The bootstrap is auto-graded against a bit-reproducible reference: you MUST
follow the seed / resample protocol exactly (treatment resampled before
control, every iteration).

    python grader.py starter.py
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

from shared import MLFPDataLoader

# --- Fixed problem constants (do not change) ---
COHORT = ["control", "treatment_a"]
BOOT_SEED = 2024            # np.random.default_rng(BOOT_SEED)
BOOT_B = 2000              # number of bootstrap resamples
MT_P_VALUES = [0.03, 0.012, 0.04, 0.65, 0.009]   # five simultaneous tests
MT_ALPHA = 0.05


def solve() -> dict:
    """Return the hypothesis-testing / bootstrap / CUPED answer dict.

    See problem.md for the exact 13 keys and how each is defined.
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp02", "experiment_data.parquet")
    co = df.filter(pl.col("experiment_group").is_in(COHORT))

    control_mask = co["experiment_group"].to_numpy() == "control"
    treatment_mask = co["experiment_group"].to_numpy() == "treatment_a"

    control = np.asarray(co.filter(pl.col("experiment_group") == "control")["metric_value"].to_numpy(), dtype=np.float64)
    treatment = np.asarray(co.filter(pl.col("experiment_group") == "treatment_a")["metric_value"].to_numpy(), dtype=np.float64)

    welch_t, welch_p = stats.ttest_ind(treatment, control, equal_var=False)
    mean_diff = float(treatment.mean() - control.mean())

    rng = np.random.default_rng(BOOT_SEED)
    diffs = np.empty(BOOT_B, dtype=np.float64)
    for b in range(BOOT_B):
        bt = rng.choice(treatment, size=treatment.size, replace=True)
        bc = rng.choice(control, size=control.size, replace=True)
        diffs[b] = bt.mean() - bc.mean()
    boot_ci_low, boot_ci_high = np.percentile(diffs, [2.5, 97.5])

    metric = np.asarray(co["metric_value"].to_numpy(), dtype=np.float64)
    pre = np.asarray(co["pre_metric_value"].to_numpy(), dtype=np.float64)
    theta = np.cov(metric, pre, ddof=1)[0, 1] / np.var(pre, ddof=1)
    pre_mean = pre.mean()
    metric_adj = metric - theta * (pre - pre_mean)
    var_metric = float(np.var(metric, ddof=1))
    var_adj = float(np.var(metric_adj, ddof=1))
    cuped_var_reduction = 1.0 - var_adj / var_metric

    control_adj = metric_adj[control_mask]
    treatment_adj = metric_adj[treatment_mask]
    welch_t_cuped, welch_p_cuped = stats.ttest_ind(treatment_adj, control_adj, equal_var=False)

    alpha = MT_ALPHA
    m = len(MT_P_VALUES)
    bonferroni_n_sig = int(sum(1 for p in MT_P_VALUES if p < alpha / m))

    sorted_p = sorted(MT_P_VALUES)
    bh_n_sig = 0
    for i, p in enumerate(sorted_p, start=1):
        threshold = alpha * i / m
        if p <= threshold:
            bh_n_sig = i

    return {
        "welch_t": float(welch_t),
        "welch_p": float(welch_p),
        "mean_diff": float(mean_diff),
        "boot_ci_low": float(boot_ci_low),
        "boot_ci_high": float(boot_ci_high),
        "cuped_theta": float(theta),
        "var_metric": float(var_metric),
        "var_adj": float(var_adj),
        "cuped_var_reduction": float(cuped_var_reduction),
        "welch_t_cuped": float(welch_t_cuped),
        "welch_p_cuped": float(welch_p_cuped),
        "bonferroni_n_sig": bonferroni_n_sig,
        "bh_n_sig": bh_n_sig,
    }


if __name__ == "__main__":
    print(solve())
