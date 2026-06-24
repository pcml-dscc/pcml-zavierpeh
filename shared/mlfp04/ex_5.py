# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP04 Exercise 5 — Association Rules.

Contains: synthetic Singapore retail transaction generator, category map,
rule dataclass helpers, and small polars utilities used by every technique
file in ``modules/mlfp04/solutions/ex_5/``.

Technique-specific code (Apriori from scratch, FP-Growth wrapper, rule
evaluation, feature engineering for classification) does NOT belong here —
it lives in the per-technique files.
"""
from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl

from kailash_ml import ExperimentTracker

from shared.kailash_helpers import setup_environment

setup_environment()

# ════════════════════════════════════════════════════════════════════════
# OUTPUT DIRECTORY
# ════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("outputs") / "mlfp04_ex5_association_rules"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# SINGAPORE RETAIL PRODUCT CATALOGUE
# ════════════════════════════════════════════════════════════════════════
# 25 products grouped to mirror a typical HDB neighbourhood mini-mart basket.

PRODUCTS: list[str] = [
    "bread",
    "butter",
    "milk",
    "eggs",
    "rice",
    "noodles",
    "soy_sauce",
    "cooking_oil",
    "chicken",
    "fish",
    "coffee",
    "tea",
    "sugar",
    "condensed_milk",
    "biscuits",
    "chips",
    "soft_drink",
    "beer",
    "wine",
    "tissue",
    "shampoo",
    "soap",
    "detergent",
    "toothpaste",
    "bananas",
]

CATEGORY_MAP: dict[str, str] = {
    "bread": "breakfast",
    "butter": "breakfast",
    "eggs": "breakfast",
    "milk": "dairy",
    "condensed_milk": "dairy",
    "coffee": "beverage",
    "tea": "beverage",
    "soft_drink": "beverage",
    "sugar": "pantry",
    "rice": "pantry",
    "cooking_oil": "pantry",
    "soy_sauce": "pantry",
    "noodles": "pantry",
    "chicken": "protein",
    "fish": "protein",
    "beer": "alcohol",
    "wine": "alcohol",
    "chips": "snack",
    "biscuits": "snack",
    "bananas": "fruit",
    "shampoo": "personal_care",
    "soap": "personal_care",
    "toothpaste": "personal_care",
    "tissue": "household",
    "detergent": "household",
}

# Co-purchase bundles (items, probability). Models real behaviour:
# kaya-toast breakfast, kopi-C, household replenishment, beer+chips, etc.
BUNDLES: list[tuple[list[str], float]] = [
    (["bread", "butter", "eggs"], 0.15),
    (["coffee", "condensed_milk", "sugar"], 0.12),
    (["rice", "chicken", "soy_sauce"], 0.10),
    (["noodles", "eggs", "soy_sauce"], 0.08),
    (["beer", "chips"], 0.09),
    (["milk", "biscuits"], 0.07),
    (["shampoo", "soap", "toothpaste"], 0.06),
    (["tea", "sugar", "biscuits"], 0.05),
    (["wine", "chips", "biscuits"], 0.04),
    (["cooking_oil", "rice", "fish"], 0.06),
    (["detergent", "tissue", "soap"], 0.05),
    (["bananas", "milk", "eggs"], 0.05),
]

N_TRANSACTIONS_DEFAULT: int = 2500


# ════════════════════════════════════════════════════════════════════════
# TRANSACTION GENERATOR
# ════════════════════════════════════════════════════════════════════════


def generate_transactions(
    n: int = N_TRANSACTIONS_DEFAULT,
    seed: int = 42,
) -> list[set[str]]:
    """Generate synthetic Singapore retail transactions.

    Each transaction is a set of product strings. Bundles fire with their
    listed probability; each item inside a firing bundle is kept with 0.85
    probability (random drop-out) so support is noisy. A Poisson number of
    random items is added on top to simulate impulse buys.
    """
    rng = np.random.default_rng(seed)
    transactions: list[set[str]] = []
    for _ in range(n):
        basket: set[str] = set()
        for bundle_items, prob in BUNDLES:
            if rng.random() < prob:
                for item in bundle_items:
                    if rng.random() < 0.85:
                        basket.add(item)
        n_random = rng.poisson(2)
        if n_random > 0:
            random_items = rng.choice(
                PRODUCTS, size=int(min(n_random, 5)), replace=False
            )
            basket.update(random_items)
        if basket:
            transactions.append(basket)
    return transactions


def transactions_to_onehot(transactions: list[set[str]]) -> pl.DataFrame:
    """One-row-per-transaction boolean matrix (columns = sorted PRODUCTS).

    Polars-native. Used as input to mlxtend FP-Growth (via .to_pandas()).
    """
    all_items = sorted(PRODUCTS)
    rows = [{item: (item in txn) for item in all_items} for txn in transactions]
    return pl.DataFrame(rows)


def product_frequency(transactions: Iterable[set[str]]) -> dict[str, int]:
    """Count how many transactions contain each product."""
    counts: dict[str, int] = defaultdict(int)
    for txn in transactions:
        for item in txn:
            counts[item] += 1
    return dict(counts)


def print_transaction_summary(transactions: list[set[str]]) -> None:
    """One-liner summary + top 10 product frequency. Used by every file."""
    avg_basket = float(np.mean([len(t) for t in transactions]))
    print("=== Synthetic Singapore Retail Transactions ===")
    print(f"  Transactions: {len(transactions):,}")
    print(f"  Products:     {len(PRODUCTS)}")
    print(f"  Avg basket:   {avg_basket:.1f} items")

    freq = product_frequency(transactions)
    n = len(transactions)
    print("\n  Top 10 products by frequency:")
    for item, count in sorted(freq.items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {item:<20} {count:>5} ({count / n:.1%})")


# ════════════════════════════════════════════════════════════════════════
# RULE HELPERS
# ════════════════════════════════════════════════════════════════════════


def format_itemset(items: Iterable[str]) -> str:
    """Deterministic pretty-print of a frozenset of items."""
    return ", ".join(sorted(items))


def categorise_rule(
    antecedent: frozenset[str],
    consequent: frozenset[str],
) -> tuple[set[str], set[str], str]:
    """Return (antecedent_categories, consequent_categories, relation_type)."""
    ant_cats = {CATEGORY_MAP.get(item, "other") for item in antecedent}
    con_cats = {CATEGORY_MAP.get(item, "other") for item in consequent}
    if ant_cats == con_cats:
        rel = "within-category complement"
    elif ant_cats & con_cats:
        rel = "cross-category with overlap"
    else:
        rel = "cross-category association"
    return ant_cats, con_cats, rel


def rules_to_polars(rules: list[dict]) -> pl.DataFrame:
    """Convert a list of rule dicts into a polars DataFrame for plotting."""
    return pl.DataFrame(
        {
            "antecedent": [format_itemset(r["antecedent"]) for r in rules],
            "consequent": [format_itemset(r["consequent"]) for r in rules],
            "support": [float(r["support"]) for r in rules],
            "confidence": [float(r["confidence"]) for r in rules],
            "lift": [float(r["lift"]) for r in rules],
        }
    )


# ════════════════════════════════════════════════════════════════════════
# KAILASH-ML EXPERIMENT TRACKER — shared by every association-rules technique
# ════════════════════════════════════════════════════════════════════════
# Every M4 ex_5 lesson logs its mining + rule-evaluation + classifier runs
# into a single SQLite store so students can compare techniques side by side
# (Apriori vs FP-Growth runtime, rule-quality at varying thresholds,
# baseline vs rule-enhanced classifier metrics) after the lesson group ends.
# Mirrors m4_clustering_zoo / m4_dimreduction_zoo / m4_anomaly_zoo from
# ex_1 / ex_3 / ex_4.
#
# IMPORTANT: this store is SEPARATE from the other M4 stores. Per the
# SQLite contention trap (.session-notes), running two ExperimentTracker
# writers against the same SQLite file concurrently triggers `disk I/O
# error`. Each exercise group gets its own DB.

ASSOC_DB = "sqlite:///mlfp04_ex5_assoc_rules.db"
ASSOC_EXPERIMENT_NAME = "m4_assoc_rules_zoo"

_TRACKER_KEY_RE = re.compile(r"[^a-zA-Z0-9_.\-]")


def _slug(text: str) -> str:
    """Coerce free-form prose into a tracker-key-safe slug.

    ExperimentTracker enforces ``^[a-zA-Z_][a-zA-Z0-9_.\\-]*$`` on metric
    names. Replace forbidden characters with underscores; ensure a leading
    alpha character; collapse repeats.
    """
    s = _TRACKER_KEY_RE.sub("_", text.strip())
    s = re.sub(r"_+", "_", s).strip("_") or "metric"
    if not s[0].isalpha() and s[0] != "_":
        s = f"m_{s}"
    return s


def _finite(x: float) -> float:
    """Tracker rejects NaN/inf via MetricValueError; coerce to 0.0.

    Association-rule metrics (lift, conviction) can collapse to inf when
    confidence saturates at 1.0; aggregate scalars over empty filter sets
    can NaN. Every emit through this guard.
    """
    return float(x) if x == x and x not in (float("inf"), float("-inf")) else 0.0


async def _setup_engines_async() -> tuple[ExperimentTracker, str]:
    """Open the association-rules ExperimentTracker (kailash-ml ≥1.5)."""
    tracker = await ExperimentTracker.create(store_url=ASSOC_DB)
    return tracker, ASSOC_EXPERIMENT_NAME


def setup_engines() -> tuple[ExperimentTracker, str]:
    """Sync wrapper. Returns (tracker, experiment_name)."""
    return asyncio.run(_setup_engines_async())


def teardown_engines(tracker: ExperimentTracker) -> None:
    """Drain the aiosqlite worker threads before the script returns.

    kailash's AsyncSQLitePool spawns NON-DAEMON aiosqlite worker threads on
    first pool use. Python 3.13's ``Py_FinalizeEx`` joins non-daemon threads
    BEFORE running ``atexit`` handlers, so an atexit-based close runs too
    late — the interpreter hangs forever in ``wait_for_thread_shutdown``
    waiting on workers stuck in ``queue.get()``.

    Solutions MUST call ``teardown_engines(tracker)`` after the REFLECTION
    block (or use it via ``try: … finally: teardown_engines(tracker)``).
    See ``rules/patterns.md`` § "Async Resource Cleanup": real cleanup is
    the caller's responsibility; ``close()`` must NOT live in ``__del__``.
    """
    asyncio.run(tracker.close())


async def _track_run_async(
    tracker: ExperimentTracker,
    exp_name: str,
    run_name: str,
    params: dict[str, Any],
    scalar_metrics: dict[str, float],
    series_metrics: dict[str, list[float]] | None = None,
) -> None:
    """Log one lesson's run: scalar metrics + optional per-step series."""
    async with tracker.track(experiment=exp_name, run_name=run_name) as run:
        await run.log_params({k: str(v) for k, v in params.items()})
        for name, value in scalar_metrics.items():
            await run.log_metric(_slug(name), _finite(value))
        if series_metrics:
            for name, values in series_metrics.items():
                slug = _slug(name)
                for step, value in enumerate(values, start=1):
                    await run.log_metric(slug, _finite(value), step=step)


def track_run(
    tracker: ExperimentTracker,
    exp_name: str,
    run_name: str,
    params: dict[str, Any],
    scalar_metrics: dict[str, float],
    series_metrics: dict[str, list[float]] | None = None,
) -> None:
    """Sync wrapper for logging a single technique's run."""
    asyncio.run(
        _track_run_async(
            tracker, exp_name, run_name, params, scalar_metrics, series_metrics
        )
    )
