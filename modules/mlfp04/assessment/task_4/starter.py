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

    trainable = SklearnTrainable(
        estimator=MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=2000,
            random_state=SEED,
        ),
        target=TARGET,
        metric="accuracy",
    )
    trainable.fit(train_df)

    test_pred = trainable.predict(test_df.select(FEATURES))
    train_pred = trainable.predict(train_df.select(FEATURES))
    test_predictions = test_pred.to_polars()[test_pred.column].to_numpy().astype(int)
    train_predictions = train_pred.to_polars()[train_pred.column].to_numpy().astype(int)

    test_labels = test_df[TARGET].to_numpy().astype(int)
    train_labels = train_df[TARGET].to_numpy().astype(int)
    test_accuracy = float((test_predictions == test_labels).mean())
    train_accuracy = float((train_predictions == train_labels).mean())

    return {
        "test_predictions": test_predictions.tolist(),
        "test_accuracy": test_accuracy,
        "train_accuracy": train_accuracy,
    }


if __name__ == "__main__":
    print(solve())
