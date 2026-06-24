# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP01 — Exercise 6: Data Visualisation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Select the appropriate chart type for a given data question
#   - Create interactive visualisations with Plotly via ModelVisualizer
#   - Apply chart design principles (Gestalt, visual hierarchy)
#   - Identify and avoid misleading chart designs
#   - Export figures as standalone HTML files for sharing
#
# PREREQUISITES: Complete Exercise 5 first (window functions, trends).
#
# ESTIMATED TIME: ~150-180 min
#
# TASKS:
#   1.  Understand ModelVisualizer and chart type selection
#   2.  Histogram — price distribution and shape analysis
#   3.  Scatter plot — price vs floor area relationships
#   4.  Bar chart — district price comparison
#   5.  Heatmap — correlation matrix between features
#   6.  Line chart — price trends over time by town
#   7.  Stacked and grouped analysis — flat type composition
#   8.  Distribution comparison — before vs after, across segments
#   9.  Gestalt principles — applying design theory to charts
#   10. Chart gallery summary and export
#
# DATASET: Singapore HDB resale flat transactions
#   Source: Housing & Development Board (data.gov.sg)
#   Rows: ~500,000 transactions | Aggregated differently per chart
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os

import numpy as np
import plotly.graph_objects as go
import polars as pl
from kailash_ml import ModelVisualizer

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────
loader = MLFPDataLoader()
hdb = loader.load("mlfp01", "hdb_resale.parquet")

# Prepare derived columns used across all charts
hdb = hdb.with_columns(
    (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
    pl.col("month").str.slice(0, 4).cast(pl.Int32).alias("year"),
    pl.col("month").str.to_date("%Y-%m").alias("transaction_date"),
)

print("=" * 60)
print("  MLFP01 Exercise 6: Data Visualisation")
print("=" * 60)
print(f"\n  Data loaded: hdb_resale.parquet ({hdb.height:,} rows, {hdb.width} columns)")
print(f"  You're ready to start!\n")

# Initialise the visualiser — one instance, many chart types
# TODO: Initialise ModelVisualizer — assign to variable `viz`
viz = ____  # Hint: ModelVisualizer()

# Create output directory for charts
os.makedirs("charts", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Understand ModelVisualizer and chart type selection
# ══════════════════════════════════════════════════════════════════════
# ModelVisualizer wraps Plotly to give you a consistent API for
# common EDA chart types. Every method returns a Plotly Figure that
# you can display inline or save as standalone HTML.
#
# CHART SELECTION GUIDE:
# ┌─────────────────────┬──────────────────────────────┐
# │ Question            │ Chart Type                    │
# ├─────────────────────┼──────────────────────────────┤
# │ Distribution shape? │ Histogram / box plot          │
# │ Two-var relation?   │ Scatter plot                  │
# │ Category comparison │ Bar chart (horiz for labels)  │
# │ Feature correlations│ Heatmap                       │
# │ Trend over time?    │ Line chart                    │
# │ Composition?        │ Stacked bar / pie             │
# │ Part of whole?      │ 100% stacked bar              │
# └─────────────────────┴──────────────────────────────┘
#
# CHARTS TO AVOID:
# - 3D charts: depth perception distorts values
# - Pie charts (>5 slices): hard to compare similar-sized slices
# - Dual y-axis: readers confuse which line uses which scale

print("=== Chart Type Reference ===")
print("  histogram()         -> Distribution shape")
print("  scatter()           -> Two-variable relationship")
print("  feature_importance() / metric_comparison() -> Category comparison")
print("  confusion_matrix()  -> Heatmap (any 2D grid)")
print("  training_history()  -> Line chart (time series)")
print("  feature_distribution() -> Single-feature distribution")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert viz is not None, "ModelVisualizer should be initialised"
print("\n✓ Checkpoint 1 passed — ModelVisualizer ready\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Histogram — price distribution
# ══════════════════════════════════════════════════════════════════════
# Histograms reveal the shape of a distribution:
# - Where is the peak? (modal price)
# - Symmetric or skewed? (long right tail = right-skewed)
# - Multiple peaks? (different market segments)

# --- 2a: Overall price distribution ---
# TODO: Call viz.histogram() with data=hdb, column="resale_price", bins=50
fig_hist = viz.histogram(
    data=____,  # Hint: hdb
    column=____,  # Hint: "resale_price"
    bins=50,
    title="HDB Resale Price Distribution (All Years)",
)
fig_hist.write_html("charts/ex6_price_histogram.html")
print("Saved: charts/ex6_price_histogram.html")
# INTERPRETATION: Right-skewed distribution — most transactions cluster
# in S$350k-600k, with a long tail of >S$800k transactions. Mean > median
# because expensive outliers pull the average upward.

# --- 2b: Price per sqm distribution ---
hdb_clean = hdb.filter(pl.col("price_per_sqm").is_not_null())
fig_sqm = viz.histogram(
    data=hdb_clean,
    column="price_per_sqm",
    bins=50,
    title="HDB Price per sqm Distribution",
)
fig_sqm.write_html("charts/ex6_price_per_sqm_histogram.html")
print("Saved: charts/ex6_price_per_sqm_histogram.html")

# --- 2c: Feature distribution with ModelVisualizer ---
clean_prices = hdb["resale_price"].drop_nulls().to_list()
fig_feat_dist = viz.histogram(
    pl.DataFrame({"Resale Price (S$)": clean_prices[:50_000]}),
    "Resale Price (S$)",
    title="Resale Price (S$)",
)
fig_feat_dist.write_html("charts/ex6_price_feature_dist.html")
print("Saved: charts/ex6_price_feature_dist.html")

# --- 2d: Compare distributions across flat types ---
# Build separate histograms for 3-room, 4-room, and 5-room
fig_compare = go.Figure()
for flat_type, color in [
    ("3 ROOM", "#636EFA"),
    ("4 ROOM", "#EF553B"),
    ("5 ROOM", "#00CC96"),
]:
    prices = hdb.filter(pl.col("flat_type") == flat_type)["resale_price"].to_list()
    fig_compare.add_trace(
        go.Histogram(
            x=prices,
            name=flat_type,
            opacity=0.6,
            marker_color=color,
            nbinsx=40,
        )
    )
fig_compare.update_layout(
    title="Price Distribution by Flat Type",
    barmode="overlay",
    xaxis_title="Resale Price (S$)",
    yaxis_title="Count",
)
fig_compare.write_html("charts/ex6_price_by_flat_type.html")
print("Saved: charts/ex6_price_by_flat_type.html")
# INTERPRETATION: Overlaid histograms show how flat type segments the market.
# 3-room flats have a tighter, lower distribution; 5-room extends further right.
# The overlap zone (S$400-600k) is where size and location trade off.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert os.path.exists("charts/ex6_price_histogram.html")
assert os.path.exists("charts/ex6_price_per_sqm_histogram.html")
assert os.path.exists("charts/ex6_price_by_flat_type.html")
print("\n✓ Checkpoint 2 passed — histogram files saved\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Scatter plot — price vs floor area
# ══════════════════════════════════════════════════════════════════════
# Scatter plots reveal relationships between two numeric variables.

# --- 3a: Basic scatter — price vs area ---
hdb_sample = hdb.sample(n=min(5_000, hdb.height), seed=42)

# TODO: Call viz.scatter() with data=hdb_sample, x="floor_area_sqm", y="resale_price"
fig_scatter = viz.scatter(
    data=hdb_sample,
    x=____,  # Hint: "floor_area_sqm"
    y=____,  # Hint: "resale_price"
    title="HDB Resale Price vs Floor Area",
)
fig_scatter.write_html("charts/ex6_price_vs_area.html")
print("Saved: charts/ex6_price_vs_area.html")
# INTERPRETATION: Positive relationship but with wide vertical spread.
# At 100 sqm, prices range from S$400k to S$900k+. This spread is
# explained by location, floor level, and remaining lease.

# --- 3b: Scatter with colour by town category ---
central_towns = ["BISHAN", "TOA PAYOH", "QUEENSTOWN", "BUKIT MERAH"]
hdb_sample = hdb_sample.with_columns(
    pl.when(pl.col("town").is_in(central_towns))
    .then(pl.lit("Central"))
    .otherwise(pl.lit("Non-Central"))
    .alias("location_type")
)

fig_scatter_color = go.Figure()
for loc_type, color in [("Central", "#EF553B"), ("Non-Central", "#636EFA")]:
    subset = hdb_sample.filter(pl.col("location_type") == loc_type)
    fig_scatter_color.add_trace(
        go.Scatter(
            x=subset["floor_area_sqm"].to_list(),
            y=subset["resale_price"].to_list(),
            mode="markers",
            name=loc_type,
            marker=dict(color=color, size=4, opacity=0.5),
        )
    )
fig_scatter_color.update_layout(
    title="Price vs Area — Central vs Non-Central",
    xaxis_title="Floor Area (sqm)",
    yaxis_title="Resale Price (S$)",
)
fig_scatter_color.write_html("charts/ex6_price_area_by_location.html")
print("Saved: charts/ex6_price_area_by_location.html")

# --- 3c: Price per sqm vs year (market appreciation) ---
annual_sample = hdb.sample(n=min(5_000, hdb.height), seed=123)
fig_year_psm = go.Figure()
fig_year_psm.add_trace(
    go.Scatter(
        x=annual_sample["year"].to_list(),
        y=annual_sample["price_per_sqm"].to_list(),
        mode="markers",
        marker=dict(size=3, opacity=0.3, color="#636EFA"),
        name="Transactions",
    )
)
fig_year_psm.update_layout(
    title="Price per sqm Over Time",
    xaxis_title="Year",
    yaxis_title="Price per sqm (S$)",
)
fig_year_psm.write_html("charts/ex6_psm_over_time.html")
print("Saved: charts/ex6_psm_over_time.html")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert os.path.exists("charts/ex6_price_vs_area.html")
assert os.path.exists("charts/ex6_price_area_by_location.html")
print("\n✓ Checkpoint 3 passed — scatter plots saved\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Bar chart — district price comparison
# ══════════════════════════════════════════════════════════════════════
# Bar charts compare a single metric across categories.
# Horizontal bars are better when labels are long (town names).

# --- 4a: Median price by town ---
district_prices = (
    hdb.group_by("town")
    .agg(
        pl.col("resale_price").median().alias("median_price"),
        pl.col("price_per_sqm").median().alias("median_price_sqm"),
        pl.len().alias("transaction_count"),
    )
    .sort("median_price", descending=True)
)

price_by_town = {
    town: {"Median Price (S$)": price}
    for town, price in zip(
        district_prices["town"].to_list(),
        district_prices["median_price"].to_list(),
    )
}
# TODO: Call viz.metric_comparison() with price_by_town dict
fig_bar = viz.metric_comparison(____)  # Hint: price_by_town
fig_bar.update_layout(title="Median HDB Price by Town")
fig_bar.write_html("charts/ex6_median_price_by_town.html")
print("Saved: charts/ex6_median_price_by_town.html")

# --- 4b: Transaction volume by town ---
volume_by_town = {
    town: {"Transactions": float(count)}
    for town, count in zip(
        district_prices.sort("transaction_count", descending=True)["town"].to_list(),
        district_prices.sort("transaction_count", descending=True)[
            "transaction_count"
        ].to_list(),
    )
}
fig_volume = viz.metric_comparison(volume_by_town)
fig_volume.update_layout(title="Transaction Volume by Town")
fig_volume.write_html("charts/ex6_volume_by_town.html")
print("Saved: charts/ex6_volume_by_town.html")

# --- 4c: Price per sqm by town (normalised comparison) ---
psm_by_town = {
    town: {"Price per sqm (S$)": float(psm)}
    for town, psm in zip(
        district_prices.sort("median_price_sqm", descending=True)["town"].to_list(),
        district_prices.sort("median_price_sqm", descending=True)[
            "median_price_sqm"
        ].to_list(),
    )
}
fig_psm = viz.metric_comparison(psm_by_town)
fig_psm.update_layout(title="Median Price per sqm by Town")
fig_psm.write_html("charts/ex6_psm_by_town.html")
print("Saved: charts/ex6_psm_by_town.html")
# INTERPRETATION: The bar chart immediately reveals the price hierarchy.
# Gestalt principle of continuity: horizontal bars are easier to compare
# than vertical ones when labels are long.

# --- 4d: Flat type comparison ---
flat_stats = (
    hdb.group_by("flat_type")
    .agg(
        pl.col("resale_price").median().alias("median_price"),
        pl.col("price_per_sqm").median().alias("median_psm"),
        pl.col("floor_area_sqm").median().alias("median_area"),
        pl.len().alias("count"),
    )
    .sort("median_price")
)

flat_metrics = {
    ft: {"Median Price": float(price), "Median PSM": float(psm)}
    for ft, price, psm in zip(
        flat_stats["flat_type"].to_list(),
        flat_stats["median_price"].to_list(),
        flat_stats["median_psm"].to_list(),
    )
}
fig_flat = viz.metric_comparison(flat_metrics)
fig_flat.update_layout(title="Price Metrics by Flat Type")
fig_flat.write_html("charts/ex6_flat_type_comparison.html")
print("Saved: charts/ex6_flat_type_comparison.html")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert os.path.exists("charts/ex6_median_price_by_town.html")
assert os.path.exists("charts/ex6_volume_by_town.html")
assert os.path.exists("charts/ex6_flat_type_comparison.html")
print("\n✓ Checkpoint 4 passed — bar charts saved\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Heatmap — correlation matrix
# ══════════════════════════════════════════════════════════════════════
# Correlation heatmaps show whether pairs of variables move together.
# +1 = perfectly positive, -1 = perfectly negative, 0 = no relationship.

# --- 5a: Build correlation matrix ---
numeric_cols = ["resale_price", "floor_area_sqm", "price_per_sqm", "year"]
hdb_numeric = hdb.select(numeric_cols).drop_nulls()

np_data = hdb_numeric.to_numpy()
corr_matrix = np.corrcoef(np_data, rowvar=False)
corr_data = [
    [round(float(corr_matrix[i, j]), 3) for j in range(len(numeric_cols))]
    for i in range(len(numeric_cols))
]

# --- 5b: Plot with Plotly heatmap ---
fig_heatmap = go.Figure(
    data=go.Heatmap(
        z=corr_data,
        x=numeric_cols,
        y=numeric_cols,
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
        text=[[str(v) for v in row] for row in corr_data],
        texttemplate="%{text}",
    )
)
fig_heatmap.update_layout(
    title="Pearson Correlation Matrix — HDB Features",
    width=650,
    height=550,
)
fig_heatmap.write_html("charts/ex6_correlation_heatmap.html")
print("Saved: charts/ex6_correlation_heatmap.html")

# Print as text table
print(f"\n=== Pearson Correlations ===")
header = f"{'':>20}" + "".join(f"{c:>16}" for c in numeric_cols)
print(header)
for col_a, row in zip(numeric_cols, corr_data):
    row_str = f"{col_a:>20}" + "".join(f"{v:>16.3f}" for v in row)
    print(row_str)
# INTERPRETATION: Diagonal is always 1.0. Off-diagonal values show:
# - resale_price vs floor_area: moderate positive (bigger = more expensive)
# - resale_price vs year: positive (prices have risen over time)
# - floor_area vs price_per_sqm: near-zero (bigger flats in cheaper towns)

# --- 5c: Extended correlation with more features ---
hdb_extended = hdb.with_columns(
    pl.col("month").str.slice(5, 2).cast(pl.Int32).alias("month_num"),
)
extended_cols = ["resale_price", "floor_area_sqm", "price_per_sqm", "year", "month_num"]
ext_numeric = hdb_extended.select(extended_cols).drop_nulls().to_numpy()
ext_corr = np.corrcoef(ext_numeric, rowvar=False)

fig_ext = go.Figure(
    data=go.Heatmap(
        z=[
            [round(float(ext_corr[i, j]), 3) for j in range(len(extended_cols))]
            for i in range(len(extended_cols))
        ],
        x=extended_cols,
        y=extended_cols,
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
        texttemplate="%{z:.3f}",
    )
)
fig_ext.update_layout(
    title="Extended Correlation Matrix",
    width=700,
    height=600,
)
fig_ext.write_html("charts/ex6_extended_correlation.html")
print("Saved: charts/ex6_extended_correlation.html")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert os.path.exists("charts/ex6_correlation_heatmap.html")
for i in range(len(numeric_cols)):
    assert abs(corr_data[i][i] - 1.0) < 0.001, f"Diagonal [{i}][{i}] should be 1.0"
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        assert abs(corr_data[i][j] - corr_data[j][i]) < 0.001, "Should be symmetric"
print("\n✓ Checkpoint 5 passed — correlation heatmap with valid symmetric matrix\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Line chart — price trends over time
# ══════════════════════════════════════════════════════════════════════
# Line charts are the natural choice for time series data.

# --- 6a: Annual median for top 5 towns ---
top_5_towns = (
    district_prices.sort("transaction_count", descending=True)["town"].head(5).to_list()
)

annual_prices = (
    hdb.filter(pl.col("town").is_in(top_5_towns))
    .group_by("year", "town")
    .agg(pl.col("resale_price").median().alias("median_price"))
    .sort("year")
)

years = sorted(annual_prices["year"].unique().to_list())
price_series: dict[str, list[float]] = {}
for town in top_5_towns:
    town_data = annual_prices.filter(pl.col("town") == town).sort("year")
    by_year = dict(
        zip(town_data["year"].to_list(), town_data["median_price"].to_list())
    )
    price_series[town] = [float(by_year.get(y, 0) or 0) for y in years]

# TODO: Call viz.training_history() with metrics=price_series, labelled axes
fig_line = viz.training_history(
    metrics=____,  # Hint: price_series
    x_label=____,  # Hint: "Year"
    y_label=____,  # Hint: "Median Resale Price (S$)"
)
fig_line.update_layout(title="Annual Median HDB Price — Top 5 Towns")
fig_line.write_html("charts/ex6_price_trends_top5.html")
print("Saved: charts/ex6_price_trends_top5.html")

# --- 6b: National trend ---
national_annual = (
    hdb.group_by("year")
    .agg(
        pl.col("resale_price").median().alias("median_price"),
        pl.col("price_per_sqm").median().alias("median_price_sqm"),
        pl.len().alias("transactions"),
    )
    .sort("year")
)

national_series = {
    "National Median Price": [
        float(v) for v in national_annual["median_price"].to_list()
    ],
}
fig_national = viz.training_history(
    metrics=national_series,
    x_label="Year",
    y_label="Median Resale Price (S$)",
)
fig_national.update_layout(title="Singapore HDB National Price Trend")
fig_national.write_html("charts/ex6_national_price_trend.html")
print("Saved: charts/ex6_national_price_trend.html")

# --- 6c: Volume trend (dual chart) ---
volume_series = {
    "Transaction Volume": [float(v) for v in national_annual["transactions"].to_list()],
}
fig_vol_trend = viz.training_history(
    metrics=volume_series,
    x_label="Year",
    y_label="Number of Transactions",
)
fig_vol_trend.update_layout(title="Annual HDB Transaction Volume")
fig_vol_trend.write_html("charts/ex6_volume_trend.html")
print("Saved: charts/ex6_volume_trend.html")

# --- 6d: Price per sqm trend (normalised) ---
psm_national = {
    "National Median PSM": [
        float(v) for v in national_annual["median_price_sqm"].to_list()
    ],
}
fig_psm_trend = viz.training_history(
    metrics=psm_national,
    x_label="Year",
    y_label="Median Price per sqm (S$)",
)
fig_psm_trend.update_layout(title="National Price per sqm Trend")
fig_psm_trend.write_html("charts/ex6_psm_trend.html")
print("Saved: charts/ex6_psm_trend.html")
# INTERPRETATION: Line charts reveal divergence between towns over time.
# The Gestalt principle of connection applies: lines make temporal
# patterns visible in a way bar charts cannot.

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert os.path.exists("charts/ex6_price_trends_top5.html")
assert os.path.exists("charts/ex6_national_price_trend.html")
assert len(top_5_towns) == 5
assert len(price_series) == 5
print("\n✓ Checkpoint 6 passed — line charts saved\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Stacked and grouped analysis — flat type composition
# ══════════════════════════════════════════════════════════════════════

# --- 7a: Flat type composition by town (stacked bar) ---
# What percentage of each town's transactions are 3-room, 4-room, etc.?
main_types = ["3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"]
top_10_towns = district_prices["town"].head(10).to_list()

composition = (
    hdb.filter(
        pl.col("town").is_in(top_10_towns) & pl.col("flat_type").is_in(main_types)
    )
    .group_by("town", "flat_type")
    .agg(pl.len().alias("count"))
)

# Build a stacked bar chart
fig_stacked = go.Figure()
colors = {
    "3 ROOM": "#636EFA",
    "4 ROOM": "#EF553B",
    "5 ROOM": "#00CC96",
    "EXECUTIVE": "#AB63FA",
}

for ft in main_types:
    ft_data = composition.filter(pl.col("flat_type") == ft)
    town_counts = dict(zip(ft_data["town"].to_list(), ft_data["count"].to_list()))
    fig_stacked.add_trace(
        go.Bar(
            name=ft,
            x=top_10_towns,
            y=[town_counts.get(t, 0) for t in top_10_towns],
            marker_color=colors.get(ft),
        )
    )

fig_stacked.update_layout(
    barmode="stack",
    title="Flat Type Composition — Top 10 Towns by Price",
    xaxis_title="Town",
    yaxis_title="Transaction Count",
)
fig_stacked.write_html("charts/ex6_flat_composition_stacked.html")
print("Saved: charts/ex6_flat_composition_stacked.html")

# --- 7b: Percentage stacked (100%) ---
fig_pct = go.Figure()
for ft in main_types:
    ft_data = composition.filter(pl.col("flat_type") == ft)
    town_counts = dict(zip(ft_data["town"].to_list(), ft_data["count"].to_list()))

    # Compute totals per town for percentage
    town_totals = {}
    for t in top_10_towns:
        total = sum(composition.filter(pl.col("town") == t)["count"].to_list())
        town_totals[t] = total

    pcts = [town_counts.get(t, 0) / town_totals.get(t, 1) * 100 for t in top_10_towns]
    fig_pct.add_trace(
        go.Bar(
            name=ft,
            x=top_10_towns,
            y=pcts,
            marker_color=colors.get(ft),
        )
    )

fig_pct.update_layout(
    barmode="stack",
    title="Flat Type Mix (%) — Top 10 Towns",
    xaxis_title="Town",
    yaxis_title="Percentage of Transactions",
)
fig_pct.write_html("charts/ex6_flat_composition_pct.html")
print("Saved: charts/ex6_flat_composition_pct.html")
# INTERPRETATION: 100% stacked bars show composition, not volume.
# Some towns are dominated by 4-room flats; others have a more even mix.
# Towns with many executive flats tend to be mature estates.

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert os.path.exists("charts/ex6_flat_composition_stacked.html")
assert os.path.exists("charts/ex6_flat_composition_pct.html")
print("\n✓ Checkpoint 7 passed — stacked bar charts saved\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Distribution comparison — segments and time periods
# ══════════════════════════════════════════════════════════════════════

# --- 8a: Box plot comparison across years ---
recent_years = [y for y in sorted(hdb["year"].unique().to_list()) if y >= 2018]
fig_box = go.Figure()
for yr in recent_years:
    yr_prices = hdb.filter(pl.col("year") == yr)["resale_price"].to_list()
    fig_box.add_trace(go.Box(y=yr_prices, name=str(yr)))

fig_box.update_layout(
    title="Price Distribution by Year (2018+)",
    yaxis_title="Resale Price (S$)",
)
fig_box.write_html("charts/ex6_box_by_year.html")
print("Saved: charts/ex6_box_by_year.html")
# INTERPRETATION: Box plots show the median (line), IQR (box), and
# outliers (dots) for each year. Rising medians confirm price appreciation;
# widening boxes show increasing price dispersion.

# --- 8b: Violin plot for flat types ---
fig_violin = go.Figure()
for ft in ["3 ROOM", "4 ROOM", "5 ROOM"]:
    prices = (
        hdb.filter(pl.col("flat_type") == ft)["resale_price"]
        .sample(n=min(10_000, hdb.filter(pl.col("flat_type") == ft).height), seed=42)
        .to_list()
    )
    fig_violin.add_trace(go.Violin(y=prices, name=ft, box_visible=True))

fig_violin.update_layout(
    title="Price Distribution by Flat Type",
    yaxis_title="Resale Price (S$)",
)
fig_violin.write_html("charts/ex6_violin_flat_type.html")
print("Saved: charts/ex6_violin_flat_type.html")

# --- 8c: Year-over-year median comparison ---
if len(recent_years) >= 2:
    first_year = recent_years[0]
    last_year = recent_years[-1]

    yr_comparison = {
        f"{first_year}": {
            "Median Price": float(
                hdb.filter(pl.col("year") == first_year)["resale_price"].median()
            ),
            "Median PSM": float(
                hdb.filter(pl.col("year") == first_year)["price_per_sqm"].median()
            ),
        },
        f"{last_year}": {
            "Median Price": float(
                hdb.filter(pl.col("year") == last_year)["resale_price"].median()
            ),
            "Median PSM": float(
                hdb.filter(pl.col("year") == last_year)["price_per_sqm"].median()
            ),
        },
    }
    fig_yr_comp = viz.metric_comparison(yr_comparison)
    fig_yr_comp.update_layout(title=f"Price Comparison: {first_year} vs {last_year}")
    fig_yr_comp.write_html("charts/ex6_year_comparison.html")
    print("Saved: charts/ex6_year_comparison.html")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert os.path.exists("charts/ex6_box_by_year.html")
assert os.path.exists("charts/ex6_violin_flat_type.html")
print("\n✓ Checkpoint 8 passed — distribution comparisons saved\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Gestalt principles — design theory in practice
# ══════════════════════════════════════════════════════════════════════
# Gestalt principles explain how humans perceive visual information.
# Applying them makes charts instantly more readable.

print("=== Gestalt Principles for Data Visualisation ===")
print(
    """
  1. PROXIMITY: Elements close together are perceived as a group.
     -> Group related bars/lines together. Separate categories with space.

  2. SIMILARITY: Elements that look alike are perceived as related.
     -> Use consistent colours for the same category across all charts.
     -> Same flat type = same colour everywhere.

  3. CLOSURE: The mind fills in missing information to complete shapes.
     -> Grid lines don't need to be solid — dotted lines work fine.
     -> Axis labels at ends; the mind fills the scale between them.

  4. ENCLOSURE: Elements inside a boundary are perceived as a group.
     -> Use subtle background shading to group related chart sections.

  5. CONTINUITY: The eye follows smooth paths over abrupt changes.
     -> Line charts leverage this — trends are visible because the eye
        follows the line. Scatter plots require more cognitive effort.

  6. CONNECTION: Connected elements are perceived as related.
     -> Lines connecting data points (time series) are more powerful
        than bars at showing temporal relationships.

  VISUAL ORDER: Z-pattern reading (left-to-right, top-to-bottom).
     -> Put the most important information top-left.
     -> Put the conclusion/summary bottom-right.
"""
)

# --- 9a: Apply principles — clean chart with annotations ---
fig_clean = go.Figure()

# Use consistent colours and grouping
national_data = national_annual.sort("year")
fig_clean.add_trace(
    go.Scatter(
        x=national_data["year"].to_list(),
        y=national_data["median_price"].to_list(),
        mode="lines+markers",
        name="Median Price",
        line=dict(color="#636EFA", width=3),
        marker=dict(size=6),
    )
)

# Add annotation for key insight (Gestalt: closure — text completes meaning)
max_price_year = national_data.sort("median_price", descending=True).row(0, named=True)
fig_clean.add_annotation(
    x=max_price_year["year"],
    y=max_price_year["median_price"],
    text=f"Peak: S${max_price_year['median_price']:,.0f}",
    showarrow=True,
    arrowhead=2,
)

# TODO: Update layout with title, axis labels, and template="plotly_white"
fig_clean.update_layout(
    title=____,  # Hint: "Singapore HDB National Price Trend"
    xaxis_title=____,  # Hint: "Year"
    yaxis_title=____,  # Hint: "Median Resale Price (S$)"
    template=____,  # Hint: "plotly_white"
)
fig_clean.write_html("charts/ex6_gestalt_clean_chart.html")
print("\nSaved: charts/ex6_gestalt_clean_chart.html")

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert os.path.exists("charts/ex6_gestalt_clean_chart.html")
print("\n✓ Checkpoint 9 passed — Gestalt principles applied\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Chart gallery summary
# ══════════════════════════════════════════════════════════════════════

# Collect all output files
chart_files = sorted(
    f for f in os.listdir("charts") if f.startswith("ex6_") and f.endswith(".html")
)

descriptions = {
    "ex6_price_histogram.html": "Overall price distribution — market shape",
    "ex6_price_per_sqm_histogram.html": "Price/sqm distribution — normalised view",
    "ex6_price_feature_dist.html": "Feature distribution via ModelVisualizer",
    "ex6_price_by_flat_type.html": "Price distribution by flat type — market segments",
    "ex6_price_vs_area.html": "Price vs area scatter — size-price relationship",
    "ex6_price_area_by_location.html": "Central vs non-central scatter",
    "ex6_psm_over_time.html": "Price per sqm over time — appreciation scatter",
    "ex6_median_price_by_town.html": "Median price by town — price hierarchy",
    "ex6_volume_by_town.html": "Transaction volume by town — market activity",
    "ex6_psm_by_town.html": "Price per sqm by town — normalised comparison",
    "ex6_flat_type_comparison.html": "Flat type price metrics",
    "ex6_correlation_heatmap.html": "Feature correlation matrix",
    "ex6_extended_correlation.html": "Extended correlation with temporal features",
    "ex6_price_trends_top5.html": "Price trends — top 5 towns by volume",
    "ex6_national_price_trend.html": "National price trend line",
    "ex6_volume_trend.html": "Transaction volume trend",
    "ex6_psm_trend.html": "National price per sqm trend",
    "ex6_flat_composition_stacked.html": "Flat type composition (stacked counts)",
    "ex6_flat_composition_pct.html": "Flat type composition (percentage)",
    "ex6_box_by_year.html": "Price distribution by year (box plots)",
    "ex6_violin_flat_type.html": "Price distribution by flat type (violin)",
    "ex6_year_comparison.html": "Year-over-year price comparison",
    "ex6_gestalt_clean_chart.html": "Clean chart with Gestalt design principles",
}

print(f"\n{'═' * 65}")
print(f"  VISUALISATION GALLERY — {len(chart_files)} Charts")
print(f"{'═' * 65}")
for filename in chart_files:
    desc = descriptions.get(filename, "")
    print(f"  {filename}")
    if desc:
        print(f"    {desc}")
print(f"{'═' * 65}")

# ── Checkpoint 10 ────────────────────────────────────────────────────
assert len(chart_files) >= 15, f"Expected at least 15 charts, got {len(chart_files)}"
missing = [f for f in chart_files if not os.path.exists(f"charts/{f}")]
assert not missing, f"Missing chart files: {missing}"
print(f"\n✓ Checkpoint 10 passed — {len(chart_files)} visualisation files saved\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 60)
print("  WHAT YOU'VE MASTERED")
print("═" * 60)
print(
    """
  ✓ ModelVisualizer: histogram, scatter, metric_comparison, training_history
  ✓ Chart selection: histogram->distribution, scatter->relationship,
    bar->comparison, heatmap->correlation, line->time series
  ✓ Plotly direct: go.Histogram, go.Scatter, go.Bar, go.Heatmap,
    go.Box, go.Violin for custom visualisations
  ✓ Overlaid histograms: comparing distributions across segments
  ✓ Coloured scatter: adding a categorical dimension to scatter plots
  ✓ Stacked bars: showing composition (counts and percentages)
  ✓ Box plots and violin plots: comparing distributions across groups
  ✓ Correlation heatmap: Pearson correlation matrix with numpy
  ✓ Chart annotations: highlighting key data points
  ✓ Gestalt principles: proximity, similarity, closure, continuity
  ✓ HTML export: fig.write_html() for shareable standalone files
  ✓ Sampling for performance: 5,000 points for scatter readability

  NEXT: In Exercise 7, you'll automate the data quality analysis
  you've been doing manually. DataExplorer profiles entire datasets
  in one call — detecting missing values, outliers, skew, correlation,
  and duplicates — and returns typed alerts that map to cleaning actions.
"""
)
