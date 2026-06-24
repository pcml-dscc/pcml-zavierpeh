# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP04 — Assessment Task 4: Neural Network Foundations

Complete the `solve()` function. Read problem.md for the full specification.
Framework-first: train the network through a kailash-ml Trainable adapter, NOT
a raw torch training loop. A multi-layer perceptron (a network with a hidden
layer) is required — a linear model cannot pass.

    python grader.py starter.py     # grade your attempt
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.neural_network import MLPClassifier

from kailash_ml import SklearnTrainable

SEED = 20260404
N = 800
SPLIT = 600  # first 600 rows train, last 200 test
FEATURES = ["x1", "x2"]
TARGET = "label"


def make_circles() -> pl.DataFrame:
    """Two concentric rings — class 0 inside, class 1 outside.

    The classes share a centre, so NO straight line separates them. Do NOT
    change the seed or sizes — the grader regenerates this exact dataset and
    the same train/test split.
    """
    rng = np.random.default_rng(SEED)
    m = N // 2

    def ring(radius: float, noise: float) -> np.ndarray:
        theta = rng.uniform(0, 2 * np.pi, m)
        r = radius + rng.normal(0, noise, m)
        return np.c_[r * np.cos(theta), r * np.sin(theta)]

    X = np.vstack([ring(1.0, 0.18), ring(3.0, 0.30)])
    y = np.r_[np.zeros(m, dtype=int), np.ones(m, dtype=int)]
    perm = rng.permutation(N)
    X, y = X[perm], y[perm]
    return pl.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "label": y})


def solve() -> dict:
    """Train an MLP through kailash-ml and beat the linear ceiling on circles."""
    df = make_circles()
    train_df = df.head(SPLIT)
    test_df = df.tail(N - SPLIT)

    # TODO 1: Build a SklearnTrainable wrapping an MLPClassifier with at least
    #         one hidden layer, e.g.
    #         estimator=MLPClassifier(hidden_layer_sizes=(32, 16),
    #             activation="relu", max_iter=2000, random_state=SEED),
    #         target=TARGET, metric="accuracy".
    # TODO 2: fit on train_df.
    # TODO 3: predict on test_df.select(FEATURES) and on train_df.select(FEATURES).
    #         The kailash-ml prediction object exposes .to_polars() and .column —
    #         pull the predicted-label column out as a numpy int array.
    # TODO 4: Compute test_accuracy and train_accuracy against the true labels.
    # TODO 5: Return {"test_predictions": [...] (length 200), "test_accuracy":
    #         float, "train_accuracy": float}.

    return {
        "test_predictions": [0] * (N - SPLIT),
        "test_accuracy": 0.0,
        "train_accuracy": 0.0,
    }


if __name__ == "__main__":
    print(solve())
