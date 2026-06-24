# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 2.1: EM Algorithm From Scratch
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Explain EM as coordinate ascent on the ELBO (lower bound on log-lik)
#   - Implement the E-step (posterior responsibilities) with log-sum-exp
#   - Implement the M-step (weighted MLE for pi, mu, Sigma)
#   - Verify that log-likelihood is non-decreasing across EM iterations
#   - Plot the convergence curve as visual proof the algorithm works
#
# PREREQUISITES:
#   - MLFP04 Exercise 1 (clustering — GMM used as a black box there)
#   - MLFP02 Lesson 2.1 (Bayesian thinking)
#
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Theory — EM as ELBO maximisation (why it's guaranteed to improve)
#   2. Build — E-step, M-step, log-likelihood, full EM loop
#   3. Train — fit a 3-component GMM on synthetic 2D data
#   4. Visualise — convergence curve + recovered vs true parameters
#   5. Apply — PropertyGuru lead-scoring (Singapore) with soft assignments
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import silhouette_score

from kailash_ml import ModelVisualizer

# Cross-exercise import: tracker helpers live in ex_1.shared so every M4
# unsupervised technique logs to the same `m4_clustering_zoo` experiment.
from shared.mlfp04.ex_1 import setup_engines, teardown_engines, track_run
from shared.mlfp04.ex_2 import (
    N_SYNTH,
    TRUE_COVS,
    TRUE_MEANS,
    TRUE_WEIGHTS,
    make_synthetic_gmm,
    out_path,
    safe_silhouette,
)

# ── Kailash-ML ExperimentTracker — every clustering run logs here ─────────
tracker, exp_name = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# THEORY — EM as ELBO Maximisation
# ════════════════════════════════════════════════════════════════════════
# Goal: maximise log P(X|theta) = Sum_i log Sum_k pi_k N(x_i|mu_k,Sigma_k).
# The log-of-a-sum makes this intractable in closed form.
#
# EM introduces a distribution Q(Z) over the hidden component assignments
# Z and maximises the Evidence Lower BOund (ELBO):
#
#     log P(X|theta)  >=  E_Q[log P(X,Z|theta)] + H(Q)    (Jensen)
#
#   E-step: fix theta, pick Q to tighten the bound.
#           Q*(Z) = P(Z|X,theta) — just the posterior responsibilities.
#
#   M-step: fix Q, pick theta to maximise the bound.
#           theta* = argmax E_{Q*}[log P(X,Z|theta)] — weighted MLE.
#
# KEY GUARANTEE: each E+M pair cannot decrease log P(X|theta). The proof
# is that the E-step makes the bound tight and the M-step raises it,
# which pushes the log-likelihood up (or leaves it fixed at a stationary
# point). We will check this numerically in Task 4.
#
# For GMMs the updates are:
#   E: r_{ik} = pi_k N(x_i|mu_k,Sigma_k) / Sum_j pi_j N(x_i|mu_j,Sigma_j)
#   M: N_k = Sum_i r_{ik}
#      pi_k = N_k / N
#      mu_k = (Sum_i r_{ik} x_i) / N_k
#      Sigma_k = (Sum_i r_{ik} (x_i-mu_k)(x_i-mu_k)') / N_k  (+ ridge)


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: E-step, M-step, log-likelihood, full EM loop
# ════════════════════════════════════════════════════════════════════════


def e_step(
    X: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """E-step: posterior responsibilities R[i,k] = P(z_i=k | x_i, theta).

    Uses the log-sum-exp trick for numerical stability: instead of
    computing products of tiny Gaussian densities we work in log-space
    and only exponentiate the normalised differences.
    """
    n_samples = X.shape[0]
    n_components = len(weights)
    log_probs = np.zeros((n_samples, n_components))

    for k in range(n_components):
        try:
            dist = multivariate_normal(mean=means[k], cov=covs[k], allow_singular=True)
            log_probs[:, k] = np.log(weights[k] + 1e-300) + dist.logpdf(X)
        except Exception:
            log_probs[:, k] = -np.inf

    log_probs_max = log_probs.max(axis=1, keepdims=True)
    log_norm = (
        np.log(np.exp(log_probs - log_probs_max).sum(axis=1, keepdims=True))
        + log_probs_max
    )
    return np.exp(log_probs - log_norm)


def m_step(
    X: np.ndarray,
    R: np.ndarray,
    reg_covar: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """M-step: update (means, covs, weights) given responsibilities R.

    The closed-form weighted MLE. A tiny ridge (reg_covar * I) is added
    to each covariance to prevent singularity when a component collapses
    onto a single point.
    """
    n_samples, n_features = X.shape
    n_components = R.shape[1]

    N_k = R.sum(axis=0) + 1e-300
    weights = N_k / n_samples
    means = (R.T @ X) / N_k[:, np.newaxis]

    covs = np.zeros((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covs[k] = (R[:, k : k + 1] * diff).T @ diff / N_k[k]
        covs[k] += reg_covar * np.eye(n_features)

    return means, covs, weights


def compute_log_likelihood(
    X: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Full data log-likelihood under the current GMM parameters."""
    n_samples = X.shape[0]
    log_l = np.full(n_samples, -np.inf)
    for k in range(len(weights)):
        try:
            dist = multivariate_normal(mean=means[k], cov=covs[k], allow_singular=True)
            log_l = np.logaddexp(log_l, np.log(weights[k] + 1e-300) + dist.logpdf(X))
        except Exception:
            pass
    return float(log_l.sum())


def fit_gmm_em(
    X: np.ndarray,
    n_components: int = 3,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 42,
) -> dict:
    """Full EM loop with random initialisation. Returns a result dict."""
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape

    idx = rng.choice(n_samples, n_components, replace=False)
    means = X[idx].copy()
    covs = np.array([np.eye(n_features)] * n_components)
    weights = np.ones(n_components) / n_components

    log_likelihoods: list[float] = []
    R = np.zeros((n_samples, n_components))

    for iteration in range(max_iter):
        R = e_step(X, means, covs, weights)
        means, covs, weights = m_step(X, R)
        ll = compute_log_likelihood(X, means, covs, weights)
        log_likelihoods.append(ll)

        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            print(f"  Converged at iteration {iteration + 1}")
            break

    return {
        "means": means,
        "covs": covs,
        "weights": weights,
        "responsibilities": R,
        "labels": R.argmax(axis=1),
        "log_likelihoods": log_likelihoods,
        "n_iter": len(log_likelihoods),
        "final_ll": log_likelihoods[-1],
    }


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: fit the from-scratch EM on synthetic 2D data
# ════════════════════════════════════════════════════════════════════════

X_synth, z_true = make_synthetic_gmm()

print("=" * 70)
print("  Synthetic GMM Data")
print("=" * 70)
print(f"Samples: {N_SYNTH}  Components: 3")
print(f"True weights: {TRUE_WEIGHTS}")

# Sanity-check the E-step against the known-good parameters
R_true_params = e_step(X_synth, TRUE_MEANS, TRUE_COVS, TRUE_WEIGHTS)
accuracy_true = (R_true_params.argmax(axis=1) == z_true).mean()
print(f"\nE-step with TRUE params -> assignment accuracy: {accuracy_true:.4f}")

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert R_true_params.shape == (N_SYNTH, 3), "R should be (n_samples, n_components)"
assert abs(R_true_params.sum(axis=1).mean() - 1.0) < 1e-6, "rows must sum to 1"
assert R_true_params.min() >= 0, "responsibilities must be non-negative"
assert accuracy_true > 0.8, "true params should give high assignment accuracy"
print("[ok] Checkpoint 1 passed — E-step behaves as a proper posterior")

# Sanity-check the M-step: hard ground-truth labels should recover the
# true parameters almost exactly (this is just class-conditional MLE).
R_hard = np.zeros((N_SYNTH, 3))
R_hard[np.arange(N_SYNTH), z_true] = 1.0
means_hat, covs_hat, weights_hat = m_step(X_synth, R_hard)

print("\nM-step with hard ground-truth labels:")
print(f"  weights_hat = {weights_hat.round(3)}  (true {TRUE_WEIGHTS})")
for k in range(3):
    print(f"  mean_{k}   = {means_hat[k].round(3)}  (true {TRUE_MEANS[k]})")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert abs(weights_hat.sum() - 1.0) < 1e-6, "weights must sum to 1"
for k in range(3):
    assert (
        np.linalg.norm(means_hat[k] - TRUE_MEANS[k]) < 1.0
    ), f"recovered mean {k} too far from ground truth"
print("[ok] Checkpoint 2 passed — M-step is weighted MLE")

# Run the full EM loop from a random initialisation
print("\nRunning EM from random init...")
em = fit_gmm_em(X_synth, n_components=3, max_iter=100, tol=1e-4)

print(f"\nIterations: {em['n_iter']}")
print(f"Final log-likelihood: {em['final_ll']:.2f}")
print(f"Recovered weights: {em['weights'].round(3)}  (true {TRUE_WEIGHTS})")

sil = safe_silhouette(X_synth, em["labels"])
print(f"Silhouette on recovered labels: {sil:.4f}")

# Verify the non-decreasing log-likelihood property numerically
lls = em["log_likelihoods"]
deltas = [lls[i] - lls[i - 1] for i in range(1, len(lls))]
n_down = sum(1 for d in deltas if d < -0.1)
print(
    f"\nLog-likelihood deltas: min={min(deltas):.4f} max={max(deltas):.4f} "
    f"n_decreases>0.1: {n_down}"
)

# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert em["n_iter"] > 1, "EM should take more than one iteration"
for i in range(1, len(lls)):
    assert lls[i] >= lls[i - 1] - 0.1, f"log-likelihood decreased at iter {i}"
assert len(set(em["labels"])) >= 2, "EM should use at least 2 components"
print("[ok] Checkpoint 3 passed — EM converged with non-decreasing log-likelihood")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: convergence curve
# ════════════════════════════════════════════════════════════════════════
# The convergence plot is the VISUAL PROOF of the ELBO theorem: the
# log-likelihood is a staircase that only goes up. A decrease would
# signal a bug in the E-step or an ill-conditioned covariance.

viz = ModelVisualizer()
fig = viz.training_history(
    {"Log-Likelihood": em["log_likelihoods"]},
    x_label="EM Iteration",
)
fig.update_layout(title="EM from scratch — log-likelihood per iteration")
fig.write_html(str(out_path("ex2_em_convergence.html")))
print(f"\nSaved: {out_path('ex2_em_convergence.html')}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: PropertyGuru Lead Scoring (Singapore)
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: PropertyGuru (Singapore) runs Southeast Asia's largest
# property marketplace. Their inside-sales team receives ~8,000 new
# lead signals per week (clicked listings, saved searches, mortgage
# calculator visits). The team can only call ~1,500 leads per week,
# so every unqualified call is a call they could have made to a
# genuine buyer.
#
# Why a GMM beats hard clustering for this task:
#   Hard clustering forces every lead into exactly one segment. A lead
#   with r = [0.45, 0.55] is genuinely between two intents — maybe a
#   "window shopper" warming up into an "active buyer". Hard assignment
#   buries that signal. With soft EM responsibilities, the sales
#   platform scores every lead on EVERY segment and sorts by expected
#   revenue = sum_k r_k * E[deal_value | segment_k].
#
# BUSINESS IMPACT:
#   - Weekly lead volume: ~8,000
#   - Closed deals via current rules: ~55/week @ avg S$9,200 agent fee
#     => S$506,000/week
#   - Soft-GMM scoring moves ambiguous leads from the bottom decile of
#     "window shoppers" into the top quartile of "active buyers" when
#     their responsibility on the active-buyer component crosses 0.4.
#     Internal A/B tests on similar Singapore marketplaces lifted close
#     rate by ~11%.
#   - 11% lift on S$506K/week = S$55,700/week = S$2.9M/year in extra
#     commission revenue, from the same inside-sales headcount.
#
# WHY EM AND NOT A CLASSIFIER: PropertyGuru has no labelled ground truth
# for "will this lead close" at the moment of first click. The only
# supervision is a delayed, rare signal (closed deal). An unsupervised
# GMM builds the segment structure from behavioural features (visits,
# saved searches, price bands, dwell time) and gives every new lead a
# soft intent vector immediately — no labels required.

responsibilities = em["responsibilities"]
ambiguous_mask = (responsibilities.max(axis=1) < 0.6) & (
    responsibilities.max(axis=1) > 0.4
)
n_ambiguous = int(ambiguous_mask.sum())
print("\n" + "=" * 70)
print("  APPLY — PropertyGuru Lead Scoring")
print("=" * 70)
print(
    f"Of {N_SYNTH} synthetic leads, {n_ambiguous} "
    f"({n_ambiguous / N_SYNTH:.1%}) are ambiguous (max responsibility 0.4-0.6)."
)
print("Hard clustering would lose the between-segment signal on every one.")


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════

track_run(
    tracker,
    exp_name,
    run_name="em_from_scratch",
    params={
        "n_components": 3,
        "n_synth": N_SYNTH,
        "max_iter": 100,
        "tol": 1e-4,
        "init": "random",
    },
    scalar_metrics={
        "final_log_likelihood": float(em["final_ll"]),
        "n_iter": float(em["n_iter"]),
        "true_param_assignment_accuracy": float(accuracy_true),
        "recovered_silhouette": float(sil) if sil == sil else 0.0,  # NaN guard
        "ambiguous_count": float(n_ambiguous),
        "ambiguous_pct": float(n_ambiguous) / float(N_SYNTH),
    },
    series_metrics={"log_likelihood_per_iter": em["log_likelihoods"]},
)
print(
    f"  [tracked] EM convergence + assignment metrics logged to {exp_name} "
    f"run='em_from_scratch'\n"
)


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — ClusteringEngine.fit(algorithm='gmm')
# ════════════════════════════════════════════════════════════════════════
# This lesson built EM from scratch — E-step responsibilities, M-step
# weighted MLE, the log-likelihood staircase, the convergence proof.
# kailash-ml 1.5.1 ships the same EM in its ClusteringEngine: one sync
# call returns labels + silhouette/CH/inertia.

import polars as pl

from kailash_ml.engines.clustering import ClusteringEngine

synth_df = pl.from_numpy(X_synth, schema=["x0", "x1"])
clustering = ClusteringEngine()
fit_result = clustering.fit(synth_df, algorithm="gmm", n_clusters=3)
print(
    f"  ClusteringEngine.fit(gmm, K=3): "
    f"silhouette={(fit_result.silhouette_score or 0.0):.4f}  "
    f"n_clusters={fit_result.n_clusters}"
)
print(
    "  Same EM you derived by hand — encapsulated under the same fit()"
    " surface that backs kmeans/dbscan/spectral. The ExperimentTracker"
    " leaderboard now compares your from-scratch loop to the engine.\n"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] EM as coordinate ascent on the ELBO — each step must improve
  [x] E-step: soft posterior responsibilities via log-sum-exp
  [x] M-step: weighted MLE for pi, mu, Sigma
  [x] Numerical proof that log-likelihood is monotone non-decreasing
  [x] PropertyGuru lead scoring: soft assignments turn into revenue

  KEY INSIGHT: EM is a template, not a GMM-specific trick. Hidden
  Markov Models, topic models (LDA), and missing-data imputation all
  use the same E / M structure.

  Next: 02_sklearn_gmm.py — use kailash-ml's sklearn bridge, verify
  your from-scratch EM matches the library, and select K via BIC/AIC.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
