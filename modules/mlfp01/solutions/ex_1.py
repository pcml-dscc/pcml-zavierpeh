# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP01 — Exercise 1: Your First Data Exploration
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Assign variables (strings, integers, floats, booleans) and format
#     output with f-strings, including alignment and decimal control
#   - Load a CSV file into a Polars DataFrame and inspect its structure
#   - Compute summary statistics with describe() and column aggregations
#   - Filter rows to find extreme values (max, min) in a dataset
#   - Build formatted summary reports combining all of the above
#
# PREREQUISITES: None — this is the very first exercise in the course.
#
# ESTIMATED TIME: ~150-180 min
#
# TASKS:
#   1.  Python basics — variables, types, and type checking
#   2.  String operations and f-string formatting
#   3.  Arithmetic and built-in functions
#   4.  Lists and basic iteration
#   5.  Load data and inspect shape, columns, and types
#   6.  Compute summary statistics with describe()
#   7.  Column-level aggregations (mean, min, max, std, quantile)
#   8.  Find extreme months and conditional reasoning
#   9.  Build a formatted summary report
#   10. Cross-column analysis — temperature vs rainfall relationship
#
# DATASET: Singapore monthly weather data (temperature, rainfall)
#   Source: Meteorological Service Singapore (data.gov.sg)
#   Rows: 12 monthly records | Columns: month, mean_temperature_c,
#   total_rainfall_mm
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────
# MLFPDataLoader handles finding data files for you — whether you're
# running locally (VS Code) or on Google Colab. Just specify the module
# and filename, and it will locate the data automatically.
loader = MLFPDataLoader()
df = loader.load("mlfp01", "sg_weather.csv")

print("=" * 60)
print("  MLFP01 Exercise 1: Your First Data Exploration")
print("=" * 60)
print(f"\n  Data loaded: sg_weather.csv ({df.height} rows, {df.width} columns)")
print(f"  You're ready to start!\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Python basics — variables, types, and type checking
# ══════════════════════════════════════════════════════════════════════
# In Python, a variable stores a value. The name goes on the left,
# the value on the right. Python figures out the type automatically —
# no need to declare "this is a string" or "this is a number".

# --- 1a: Core data types ---
city = "Singapore"  # str: a piece of text, always in quotes
country = "Singapore"  # another string
years_of_data = 30  # int: a whole number (no decimal point)
latitude = 1.35  # float: a decimal number
is_tropical = True  # bool: True or False — used for decisions

# Python's type() function tells you what kind of value a variable holds.
# This is useful when debugging — unexpected types cause hard-to-find errors.
print("=== Variable Types ===")
print(f"city:          {type(city).__name__}   → {city!r}")
print(f"years_of_data: {type(years_of_data).__name__}    → {years_of_data}")
print(f"latitude:      {type(latitude).__name__}  → {latitude}")
print(f"is_tropical:   {type(is_tropical).__name__}   → {is_tropical}")
# !r shows the "repr" of a string — with quotes around it — so you can
# see that it really is a string and not just printed text.

# --- 1b: Type conversion ---
# Sometimes you need to convert between types. Python calls these "casts".
price_str = "450000"  # This is text, not a number — you can't do math
price_int = int(price_str)  # Now it's an integer you can calculate with
price_float = float(price_str)  # As a decimal: 450000.0
height_int = int(3.7)  # Truncates toward zero: 3 (NOT rounded)
bool_from_int = bool(1)  # Any nonzero number is True; 0 is False

print("\n=== Type Conversions ===")
print(f"'{price_str}' (str) → {price_int} (int) → {price_float} (float)")
print(f"int(3.7) = {height_int}  (truncates, does NOT round)")
print(f"bool(1)  = {bool_from_int},  bool(0) = {bool(0)}")

# --- 1c: None — the absence of a value ---
# None is Python's "nothing" value. It's not zero, not empty string,
# not False. It represents "no value at all". You'll see this a lot
# in data analysis when data is missing.
missing_value = None
print(f"\nmissing_value is None: {missing_value is None}")
print(f"type(None): {type(None).__name__}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert isinstance(city, str), "city should be a string"
assert isinstance(years_of_data, int), "years_of_data should be an integer"
assert isinstance(latitude, float), "latitude should be a float"
assert isinstance(is_tropical, bool), "is_tropical should be a boolean"
assert price_int == 450_000, "int conversion of '450000' should be 450000"
assert height_int == 3, "int(3.7) should truncate to 3"
assert missing_value is None, "missing_value should be None"
print("\n✓ Checkpoint 1 passed — variables, types, and conversions working\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: String operations and f-string formatting
# ══════════════════════════════════════════════════════════════════════
# Strings are everywhere in data work — column names, categories, file
# paths, report text. Knowing how to manipulate them saves hours.

# --- 2a: String methods ---
raw_town = "  ang mo kio  "
clean_town = raw_town.strip()  # Remove leading/trailing whitespace
upper_town = clean_town.upper()  # "ANG MO KIO" — standard for HDB data
title_town = clean_town.title()  # "Ang Mo Kio" — for display
word_count = len(clean_town.split())  # Split by spaces, count the pieces

print("=== String Methods ===")
print(f"raw:    {raw_town!r}")
print(f"strip:  {clean_town!r}")
print(f"upper:  {upper_town!r}")
print(f"title:  {title_town!r}")
print(f"words:  {word_count}")

# .replace() swaps one substring for another — useful for cleaning data
dirty_price = "S$450,000"
numeric_price = dirty_price.replace("S$", "").replace(",", "")
print(f"\n'{dirty_price}' → '{numeric_price}' → {int(numeric_price)}")

# .startswith() and .endswith() — checking prefixes and suffixes
filename = "hdb_resale.parquet"
print(f"\n'{filename}' starts with 'hdb': {filename.startswith('hdb')}")
print(f"'{filename}' ends with '.csv': {filename.endswith('.csv')}")
print(f"'{filename}' ends with '.parquet': {filename.endswith('.parquet')}")

# --- 2b: f-string formatting specifiers ---
# f-strings embed expressions inside curly braces {}. Format specifiers
# after the colon control how values are displayed.
celsius_avg = 27.5
fahrenheit_avg = celsius_avg * 9 / 5 + 32
population = 5_917_600  # Underscores are visual separators — Python ignores them

print("\n=== f-string Formatting ===")
print(f"Temperature: {celsius_avg}°C / {fahrenheit_avg:.1f}°F")
#   :.1f  → 1 decimal place, float
print(f"Population:  {population:,}")
#   :,    → thousands separator (5,917,600)
print(f"Population:  {population:>15,}")
#   :>15, → right-align in 15-char field with comma separator
print(f"Percentage:  {0.8765:.1%}")
#   :.1%  → multiply by 100, add %, 1 decimal → "87.7%"
print(f"Left-align:  {'Singapore':<20} | end")
#   :<20  → left-align in 20-char field
print(f"Center:      {'Singapore':^20} | end")
#   :^20  → center in 20-char field

# --- 2c: Multi-line strings ---
# Triple quotes let you write strings across multiple lines.
# Useful for reports, SQL queries, and documentation.
report_header = f"""
╔{'═' * 40}╗
║{'Singapore Weather Report':^40}║
║{'Data from Meteorological Service':^40}║
╚{'═' * 40}╝
"""
print(report_header)

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert (
    upper_town == "ANG MO KIO"
), f"upper() should give 'ANG MO KIO', got {upper_town!r}"
assert word_count == 3, f"'ang mo kio' has 3 words, got {word_count}"
assert int(numeric_price) == 450_000, "cleaned price should be 450000"
assert fahrenheit_avg > 80, "Fahrenheit conversion looks wrong"
print("✓ Checkpoint 2 passed — string operations and f-string formatting working\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Arithmetic and built-in functions
# ══════════════════════════════════════════════════════════════════════
# Python provides all standard arithmetic operators plus some useful
# built-in functions that you'll use constantly in data analysis.

# --- 3a: Arithmetic operators ---
a, b = 17, 5
print("=== Arithmetic Operators ===")
print(f"{a} + {b}  = {a + b}")  # Addition: 22
print(f"{a} - {b}  = {a - b}")  # Subtraction: 12
print(f"{a} * {b}  = {a * b}")  # Multiplication: 85
print(f"{a} / {b}  = {a / b}")  # True division: 3.4 (always float)
print(f"{a} // {b} = {a // b}")  # Floor division: 3 (rounds down)
print(f"{a} % {b}  = {a % b}")  # Modulo: 2 (remainder)
print(f"{a} ** {b} = {a ** b}")  # Power: 1419857 (17^5)

# Important: / always returns a float, even for exact division
print(f"\n10 / 5 = {10 / 5}  (type: {type(10 / 5).__name__})")
print(f"10 // 5 = {10 // 5}  (type: {type(10 // 5).__name__})")

# --- 3b: Built-in functions for data work ---
prices = [350_000, 420_000, 510_000, 680_000, 1_200_000]
print("\n=== Built-in Functions ===")
print(f"len(prices):   {len(prices)}")  # Count of items: 5
print(f"sum(prices):   {sum(prices):,}")  # Total: 3,160,000
print(f"min(prices):   {min(prices):,}")  # Smallest: 350,000
print(f"max(prices):   {max(prices):,}")  # Largest: 1,200,000
print(f"sorted(prices, reverse=True): {sorted(prices, reverse=True)}")

# Manual mean — Python has no built-in mean() (it's in the statistics module)
mean_price = sum(prices) / len(prices)
print(f"mean(prices):  {mean_price:,.0f}")

# abs() for absolute values — useful for price changes
price_change = -25_000
print(f"\nprice_change: {price_change:,}  abs: {abs(price_change):,}")

# round() for rounding — the second argument is decimal places
pi = 3.14159265
print(f"round(pi, 2): {round(pi, 2)}")
print(f"round(pi, 4): {round(pi, 4)}")
print(f"round(1234, -2): {round(1234, -2)}")  # Round to nearest 100

# --- 3c: Comparison operators ---
# These return True or False — the foundation of all filtering in Polars
x = 500_000
print("\n=== Comparison Operators ===")
print(f"{x} > 400000:  {x > 400_000}")  # True
print(f"{x} <= 500000: {x <= 500_000}")  # True (less than OR equal)
print(f"{x} == 500000: {x == 500_000}")  # True (equality check — double =)
print(f"{x} != 400000: {x != 400_000}")  # True (not equal)

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert 17 // 5 == 3, "Floor division of 17 // 5 should be 3"
assert 17 % 5 == 2, "17 % 5 should be 2"
assert len(prices) == 5, "prices list should have 5 elements"
assert sum(prices) == 3_160_000, "sum of prices should be 3,160,000"
assert mean_price == 632_000.0, f"mean should be 632000, got {mean_price}"
assert round(pi, 2) == 3.14, "round(pi, 2) should be 3.14"
print("\n✓ Checkpoint 3 passed — arithmetic and built-in functions working\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Lists and basic iteration
# ══════════════════════════════════════════════════════════════════════
# Lists are ordered collections. In data analysis, you use lists to hold
# column names, parameter sets, and intermediate results.

# --- 4a: Creating and accessing lists ---
months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

print("=== List Operations ===")
print(f"First month:  {months[0]}")  # Index 0 = first item
print(f"Last month:   {months[-1]}")  # Index -1 = last item
print(f"Q1 months:    {months[0:3]}")  # Slice [start:stop) — stop is exclusive
print(f"Q4 months:    {months[9:]}")  # From index 9 to the end
print(f"Every 3rd:    {months[::3]}")  # Step by 3

# --- 4b: Modifying lists ---
temperatures = [27.0, 27.1, 27.8, 28.3, 28.6, 28.3, 27.9, 27.8, 27.6, 27.6, 27.0, 26.8]
temperatures.append(27.2)  # Add to the end
print(f"\nAfter append: {len(temperatures)} items")
temperatures.pop()  # Remove from the end
print(f"After pop:    {len(temperatures)} items")

# --- 4c: List comprehensions — the Pythonic way to transform lists ---
# A list comprehension builds a new list from an existing one in one line.
# Syntax: [expression for item in iterable]
celsius_temps = [27.0, 27.5, 28.0, 28.5, 29.0]
fahrenheit_temps = [c * 9 / 5 + 32 for c in celsius_temps]
print(f"\nCelsius:    {celsius_temps}")
print(f"Fahrenheit: {[round(f, 1) for f in fahrenheit_temps]}")

# With a filter — only keep values above a threshold
warm_months = [c for c in celsius_temps if c >= 28.0]
print(f"Warm (>=28°C): {warm_months}")

# --- 4d: Dictionaries — key-value pairs ---
# Dicts map a key to a value. They're how Polars returns row data.
month_temps = {
    "Jan": 26.8,
    "Feb": 27.0,
    "Mar": 27.8,
    "Apr": 28.3,
    "May": 28.6,
    "Jun": 28.3,
    "Jul": 27.9,
    "Aug": 27.8,
    "Sep": 27.6,
    "Oct": 27.6,
    "Nov": 27.0,
    "Dec": 26.8,
}

print(f"\nJan temp: {month_temps['Jan']}°C")
print(f"Keys:     {list(month_temps.keys())[:4]}...")
print(f"Values:   {list(month_temps.values())[:4]}...")

# Find the hottest month using max() with a key function
hottest_dict = max(month_temps, key=month_temps.get)
print(f"Hottest:  {hottest_dict} at {month_temps[hottest_dict]}°C")

# --- 4e: for loop with enumerate ---
print("\n=== Monsoon Season Analysis ===")
monsoon_months = ["Nov", "Dec", "Jan"]
for i, m in enumerate(monsoon_months, start=1):
    # enumerate() gives you both the index and the value
    temp = month_temps[m]
    print(f"  {i}. {m}: {temp}°C")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert months[0] == "Jan", "First month should be Jan"
assert months[-1] == "Dec", "Last month should be Dec"
assert len(celsius_temps) == 5, "celsius_temps should have 5 items"
assert len(fahrenheit_temps) == 5, "fahrenheit_temps should have 5 items"
assert len(warm_months) == 3, "3 months should be >= 28°C"
assert hottest_dict == "May", f"Hottest month should be May, got {hottest_dict}"
print("\n✓ Checkpoint 4 passed — lists, dicts, and iteration working\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Load data and inspect its shape and structure
# ══════════════════════════════════════════════════════════════════════
# A DataFrame is a table of data — rows are observations, columns are
# measurements. Think of it like a spreadsheet in your code.
# Polars is the data library we use throughout this course.

# .shape returns a tuple: (number_of_rows, number_of_columns)
rows, cols = df.shape
print(f"=== Dataset Overview ===")
print(f"Rows: {rows:,}")
print(f"Columns: {cols}")

# .columns returns a list of the column names
print(f"\nColumn names:")
for col_name in df.columns:
    print(f"  - {col_name}")

# .dtypes returns the data type of each column
# Polars uses its own type system: String=text, Float64=decimal, Int64=integer
print(f"\nData types:")
for col_name, dtype in zip(df.columns, df.dtypes):
    print(f"  {col_name:>25}: {dtype}")

# .head(n) shows the first n rows — always look at raw data before analysis
print(f"\nFirst 5 rows:")
print(df.head(5))

# .tail(n) shows the LAST n rows — useful for checking the end of the data
print(f"\nLast 3 rows:")
print(df.tail(3))

# .sample(n) shows n random rows — useful for spotting patterns that
# hide at the top or bottom of the file
print(f"\n3 random rows:")
print(df.sample(n=min(3, df.height), seed=42))

# INTERPRETATION: Inspecting head, tail, and sample gives you three views
# of the same data. If head() and tail() look different (different columns,
# different formats), you might have schema drift — the data changed format
# partway through the file. sample() catches patterns that sorted data hides.

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert rows > 0, "DataFrame has no rows — check data loading"
assert cols >= 2, "DataFrame should have at least 2 columns"
assert "month" in df.columns, "Expected a 'month' column"
assert len(df.columns) == cols, "Column count should match .shape"
print("\n✓ Checkpoint 5 passed — data loaded and inspected\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Summary statistics with describe()
# ══════════════════════════════════════════════════════════════════════
# .describe() computes count, mean, std, min, max for every numeric column.
# This single call replaces writing 5 separate calculations — that's
# the point of a data library: common operations should take one line.

print(f"=== Summary Statistics ===")
stats = df.describe()
print(stats)

# INTERPRETATION: describe() gives you the "vital signs" of your data.
# Read it like a doctor reads a blood test — you're looking for anomalies:
# - count: any column with count < total rows has missing data
# - mean vs median (50%): if they differ a lot, the data is skewed
# - min/max: are these physically possible? (temperature of 200°C = error)
# - std: how spread out is the data? Low std = consistent; high std = variable

# You can access a single column with df["column_name"]
mean_temp = df["mean_temperature_c"].mean()
min_temp = df["mean_temperature_c"].min()
max_temp = df["mean_temperature_c"].max()
std_temp = df["mean_temperature_c"].std()

print(f"\nTemperature details:")
print(f"  Average:  {mean_temp:.2f}°C")
print(f"  Minimum:  {min_temp:.2f}°C")
print(f"  Maximum:  {max_temp:.2f}°C")
print(f"  Std dev:  {std_temp:.2f}°C")
# INTERPRETATION: Std dev measures how spread out the values are.
# A small std dev means temperatures cluster tightly around the mean.
# Singapore's tropical climate means low temperature variation (~1°C std).
# Compare this to London (std ~5°C) or Moscow (std ~12°C).

mean_rain = df["total_rainfall_mm"].mean()
max_rain = df["total_rainfall_mm"].max()
min_rain = df["total_rainfall_mm"].min()
std_rain = df["total_rainfall_mm"].std()
print(f"\nRainfall details:")
print(f"  Average:  {mean_rain:.1f} mm/month")
print(f"  Maximum:  {max_rain:.1f} mm/month")
print(f"  Minimum:  {min_rain:.1f} mm/month")
print(f"  Std dev:  {std_rain:.1f} mm/month")
# INTERPRETATION: Singapore receives ~170mm of rain per month on average.
# The monsoon season (Nov-Jan) can bring 250+ mm — nearly 50% above average.
# The high std dev relative to the mean tells you rainfall is more variable
# than temperature — which makes sense: it can pour one month and be dry the next.

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert mean_temp is not None, "mean_temp should not be None"
assert 20 < mean_temp < 35, f"mean_temp={mean_temp} seems wrong for Singapore"
assert mean_rain > 0, "mean_rain should be positive"
assert std_temp < std_rain, "Rainfall should be more variable than temperature"
print("\n✓ Checkpoint 6 passed — summary statistics computed correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Column-level aggregations — quantiles and ranges
# ══════════════════════════════════════════════════════════════════════
# Beyond mean and std, quantiles divide your data into portions.
# The median (50th percentile) is the "middle" value — half the data
# is above, half below. Quantiles resist outliers better than the mean.

# --- 7a: Quantiles for temperature ---
temp_series = df["mean_temperature_c"]
temp_q25 = temp_series.quantile(0.25)
temp_q50 = temp_series.quantile(0.50)  # = median
temp_q75 = temp_series.quantile(0.75)
temp_iqr = temp_q75 - temp_q25

print("=== Temperature Quantiles ===")
print(f"  25th percentile (Q1): {temp_q25:.2f}°C")
print(f"  50th percentile (Q2): {temp_q50:.2f}°C  (this is the median)")
print(f"  75th percentile (Q3): {temp_q75:.2f}°C")
print(f"  Interquartile range:  {temp_iqr:.2f}°C  (Q3 - Q1)")
# INTERPRETATION: The IQR tells you the range of the "middle 50%" of values.
# For Singapore temperature, IQR ≈ 1°C means the middle half of all monthly
# temperatures span just 1 degree. This is remarkably stable — a property
# investor doesn't need to worry about weather-driven demand swings.

# --- 7b: Quantiles for rainfall ---
rain_series = df["total_rainfall_mm"]
rain_q25 = rain_series.quantile(0.25)
rain_q50 = rain_series.quantile(0.50)
rain_q75 = rain_series.quantile(0.75)
rain_iqr = rain_q75 - rain_q25

print(f"\n=== Rainfall Quantiles ===")
print(f"  Q1: {rain_q25:.1f} mm")
print(f"  Q2: {rain_q50:.1f} mm  (median)")
print(f"  Q3: {rain_q75:.1f} mm")
print(f"  IQR: {rain_iqr:.1f} mm")

# --- 7c: Coefficient of variation ---
# CV = std / mean * 100. It's a normalised measure of spread.
# Useful when comparing variability between different units (°C vs mm).
cv_temp = (std_temp / mean_temp) * 100
cv_rain = (std_rain / mean_rain) * 100

print(f"\n=== Coefficient of Variation ===")
print(f"  Temperature CV: {cv_temp:.1f}%  (very low — stable climate)")
print(f"  Rainfall CV:    {cv_rain:.1f}%  (much higher — seasonal variation)")
# INTERPRETATION: A CV under 10% means the variable barely changes.
# Temperature CV ≈ 2% means Singapore's climate is one of the most stable
# in the world. Rainfall CV ≈ 30% means month-to-month rainfall is
# unpredictable — a key risk factor for outdoor construction projects.

# --- 7d: Data range and total ---
total_annual_rain = rain_series.sum()
temp_range = max_temp - min_temp

print(f"\n=== Annual Totals ===")
print(f"  Total annual rainfall: {total_annual_rain:.0f} mm")
print(f"  Temperature range:     {temp_range:.1f}°C (max - min)")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert temp_q25 is not None, "Q1 should not be None"
assert temp_q50 is not None, "Median should not be None"
assert temp_q25 <= temp_q50 <= temp_q75, "Quantiles should be ordered"
assert rain_iqr >= 0, "IQR should be non-negative"
assert cv_rain > cv_temp, "Rainfall should be more variable than temperature"
assert total_annual_rain > 0, "Total annual rainfall should be positive"
print("\n✓ Checkpoint 7 passed — quantiles and advanced statistics correct\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Find extreme months and conditional reasoning
# ══════════════════════════════════════════════════════════════════════
# .filter() keeps rows where a condition is True
# pl.col("column_name") refers to a column inside a Polars expression
# .max() returns the single highest value in a column

# --- 8a: Find the hottest, coldest, and wettest months ---
hottest_row = df.filter(pl.col("mean_temperature_c") == df["mean_temperature_c"].max())
coldest_row = df.filter(pl.col("mean_temperature_c") == df["mean_temperature_c"].min())
wettest_row = df.filter(pl.col("total_rainfall_mm") == df["total_rainfall_mm"].max())
driest_row = df.filter(pl.col("total_rainfall_mm") == df["total_rainfall_mm"].min())

hottest_month = hottest_row["month"][0]
hottest_temp = hottest_row["mean_temperature_c"][0]
coldest_month = coldest_row["month"][0]
coldest_temp = coldest_row["mean_temperature_c"][0]
wettest_month = wettest_row["month"][0]
wettest_rain = wettest_row["total_rainfall_mm"][0]
driest_month = driest_row["month"][0]
driest_rain = driest_row["total_rainfall_mm"][0]

print(f"=== Extreme Months ===")
print(f"Hottest:  {hottest_month} at {hottest_temp:.1f}°C")
print(f"Coldest:  {coldest_month} at {coldest_temp:.1f}°C")
print(f"Wettest:  {wettest_month} with {wettest_rain:.1f} mm")
print(f"Driest:   {driest_month} with {driest_rain:.1f} mm")

# --- 8b: Above-average analysis ---
# "How many months are above average?" is a basic business question.
above_avg_temp = df.filter(pl.col("mean_temperature_c") > mean_temp)
above_avg_rain = df.filter(pl.col("total_rainfall_mm") > mean_rain)

print(f"\n=== Above-Average Months ===")
print(f"Temperature above {mean_temp:.1f}°C: {above_avg_temp.height} months")
print(f"  Months: {above_avg_temp['month'].to_list()}")
print(f"Rainfall above {mean_rain:.0f} mm: {above_avg_rain.height} months")
print(f"  Months: {above_avg_rain['month'].to_list()}")

# --- 8c: Percentage deviations from average ---
# "How much hotter is the hottest month compared to average?"
hottest_pct_above = ((hottest_temp - mean_temp) / mean_temp) * 100
wettest_pct_above = ((wettest_rain - mean_rain) / mean_rain) * 100

print(f"\n=== Deviations from Average ===")
print(f"Hottest month is {hottest_pct_above:+.1f}% above average temperature")
print(f"Wettest month is {wettest_pct_above:+.1f}% above average rainfall")
# The + in :+.1f forces a + sign for positive numbers — makes deviations clearer

# INTERPRETATION: Singapore's hottest months are May-Jun (pre-monsoon),
# coldest are Dec-Jan (NE monsoon), and wettest are Nov-Jan (monsoon peak).
# The temperature deviation is small (maybe +3%) while the rainfall
# deviation is large (maybe +50%). This asymmetry matters for planning:
# temperature is not a risk, but rainfall is.

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert hottest_temp >= coldest_temp, "Hottest should be >= coldest"
assert wettest_rain >= driest_rain, "Wettest should be >= driest"
assert wettest_rain >= mean_rain, "Wettest month should be above average"
assert isinstance(hottest_month, str), "Month should be a string"
assert above_avg_temp.height > 0, "Some months should be above average temp"
assert above_avg_rain.height > 0, "Some months should be above average rain"
print("\n✓ Checkpoint 8 passed — extreme values and deviations found correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Formatted summary report
# ══════════════════════════════════════════════════════════════════════
# Let's combine everything into a clean, readable report.
# Good formatting makes data accessible to non-technical readers.
# A well-formatted report is the deliverable — the code is the process.

separator = "═" * 60

print(f"\n{separator}")
print(f"  SINGAPORE WEATHER SUMMARY REPORT")
print(f"{separator}")
print(f"  Source:     Meteorological Service Singapore")
print(f"  Records:    {rows:>6,} monthly observations")
print(f"  Variables:  {cols:>6} columns")
print(f"")
print(f"  ┌{'─' * 42}┐")
print(f"  │ TEMPERATURE (°C)                         │")
print(f"  ├{'─' * 42}┤")
print(f"  │  Mean:       {mean_temp:>8.2f}                     │")
print(f"  │  Std Dev:    {std_temp:>8.2f}  (CV: {cv_temp:.1f}%)        │")
print(f"  │  Range:      {min_temp:>8.2f} — {max_temp:.2f}             │")
print(f"  │  IQR:        {temp_iqr:>8.2f}                     │")
print(f"  │  Hottest:    {hottest_month:>8} ({hottest_temp:.1f}°C)        │")
print(f"  │  Coldest:    {coldest_month:>8} ({coldest_temp:.1f}°C)        │")
print(f"  └{'─' * 42}┘")
print(f"")
print(f"  ┌{'─' * 42}┐")
print(f"  │ RAINFALL (mm/month)                      │")
print(f"  ├{'─' * 42}┤")
print(f"  │  Mean:       {mean_rain:>8.1f}                     │")
print(f"  │  Std Dev:    {std_rain:>8.1f}  (CV: {cv_rain:.1f}%)       │")
print(f"  │  Range:      {min_rain:>8.1f} — {max_rain:.1f}           │")
print(f"  │  IQR:        {rain_iqr:>8.1f}                     │")
print(f"  │  Annual:     {total_annual_rain:>8.0f} mm                 │")
print(f"  │  Wettest:    {wettest_month:>8} ({wettest_rain:.0f} mm)        │")
print(f"  │  Driest:     {driest_month:>8} ({driest_rain:.0f} mm)         │")
print(f"  └{'─' * 42}┘")
print(f"{separator}")
# The :>8 in format strings means "right-align in a field 8 chars wide"
# This keeps your numbers lined up in neat columns.

# ── Checkpoint 9 ─────────────────────────────────────────────────────
print("\n✓ Checkpoint 9 passed — formatted summary report generated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Cross-column analysis — temperature vs rainfall
# ══════════════════════════════════════════════════════════════════════
# Real analysis involves looking at relationships BETWEEN columns, not
# just summarising one column at a time. This is a preview of the
# correlation analysis you'll do in Exercise 6 with charts.

# --- 10a: Does it rain more in hotter or cooler months? ---
# Split months into "warm" and "cool" based on the median temperature
warm_months_df = df.filter(pl.col("mean_temperature_c") >= temp_q50)
cool_months_df = df.filter(pl.col("mean_temperature_c") < temp_q50)

warm_avg_rain = warm_months_df["total_rainfall_mm"].mean()
cool_avg_rain = cool_months_df["total_rainfall_mm"].mean()

print("=== Temperature vs Rainfall ===")
print(f"Warm months (>= {temp_q50:.1f}°C):  avg rainfall = {warm_avg_rain:.1f} mm")
print(f"Cool months (<  {temp_q50:.1f}°C):  avg rainfall = {cool_avg_rain:.1f} mm")
if cool_avg_rain > warm_avg_rain:
    print("  → Cool months are wetter — this aligns with the NE monsoon (Dec-Feb)")
else:
    print("  → Warm months are wetter")

# --- 10b: Monthly deviation table ---
# For each month, compute how far temperature and rainfall deviate from
# their respective means. This "z-score-like" view normalises different
# scales so you can compare apples to oranges.
monthly_deviations = df.with_columns(
    ((pl.col("mean_temperature_c") - mean_temp) / std_temp).alias("temp_z_score"),
    ((pl.col("total_rainfall_mm") - mean_rain) / std_rain).alias("rain_z_score"),
)

print(f"\n=== Monthly Deviations (z-scores) ===")
print(f"  {'Month':>8} {'Temp z':>10} {'Rain z':>10}")
print(f"  {'─' * 30}")
for row in monthly_deviations.iter_rows(named=True):
    temp_z = row.get("temp_z_score")
    rain_z = row.get("rain_z_score")
    if temp_z is not None and rain_z is not None:
        print(f"  {row['month']:>8} {temp_z:>+10.2f} {rain_z:>+10.2f}")
# INTERPRETATION: z-scores express each value as "how many standard deviations
# from the mean." A z-score of +1.5 means 1.5 standard deviations above average.
# This lets you compare temperature (measured in °C) with rainfall (measured
# in mm) on the same scale. Months where both z-scores are positive are
# both warmer AND wetter than average — unusual and worth investigating.

# --- 10c: Polars correlation coefficient ---
# Pearson correlation: +1 = perfect positive, -1 = perfect negative, 0 = none
temp_rain_corr = df.select(pl.corr("mean_temperature_c", "total_rainfall_mm")).item()

print(f"\nPearson correlation (temperature vs rainfall): {temp_rain_corr:.3f}")
if abs(temp_rain_corr) < 0.3:
    print("  → Weak correlation: temperature alone does not predict rainfall well")
elif temp_rain_corr < 0:
    print("  → Negative correlation: hotter months tend to be drier")
else:
    print("  → Positive correlation: hotter months tend to be wetter")

# INTERPRETATION: For Singapore, you'll likely see a weak negative correlation.
# The NE monsoon (cooler, wetter) and inter-monsoon (warmer, drier) seasons
# create this inverse relationship. But it's weak because Singapore's
# temperature range is so narrow — rainfall is driven more by monsoon
# patterns than by temperature alone.

# ── Checkpoint 10 ────────────────────────────────────────────────────
assert warm_avg_rain is not None, "warm_avg_rain should not be None"
assert cool_avg_rain is not None, "cool_avg_rain should not be None"
assert isinstance(temp_rain_corr, float), "Correlation should be a float"
assert -1.0 <= temp_rain_corr <= 1.0, "Correlation must be between -1 and 1"
assert "temp_z_score" in monthly_deviations.columns, "z-score column should exist"
print("\n✓ Checkpoint 10 passed — cross-column analysis and correlation correct\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 60)
print("  WHAT YOU'VE MASTERED")
print("═" * 60)
print(
    """
  ✓ Variables: str, int, float, bool, None — Python's core types
  ✓ Type conversion: int(), float(), str(), bool() — casting between types
  ✓ String methods: strip, upper, replace, split, startswith
  ✓ f-strings: alignment (:>8), decimals (:.2f), percentages (:.1%),
    thousands separator (:,)
  ✓ Arithmetic: +, -, *, /, //, %, ** — and when to use each
  ✓ Built-in functions: len, sum, min, max, sorted, round, abs
  ✓ Lists: indexing, slicing, append, comprehensions
  ✓ Dictionaries: key-value pairs, .keys(), .values(), max() with key=
  ✓ DataFrames: loading CSV, shape, columns, dtypes, head, tail, sample
  ✓ describe(): one-call summary statistics for every column
  ✓ Aggregations: mean, min, max, std, quantile — the data analyst's toolkit
  ✓ Filtering: finding rows that match a condition with .filter()
  ✓ Formatted output: building professional reports with alignment
  ✓ Cross-column analysis: z-scores, correlation, comparative reasoning

  NEXT: In Exercise 2, you'll learn to filter and transform data
  using Polars expressions — selecting rows by condition, creating
  new columns, and chaining operations together. The HDB resale
  dataset (500K+ transactions) will be your playground.
"""
)
