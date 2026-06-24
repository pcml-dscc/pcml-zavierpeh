# MLFP01 — Task 1: Taxi Trip Data Forensics

**Weight**: 25 marks · **Difficulty**: Hard · **Dataset**: `data/mlfp01/sg_taxi_trips.parquet` (50,000 raw rows, 12 columns)

## Scenario

A Singapore ride-hailing operator hands you a raw trip log straight from three
merged dispatch systems. It is dirty: duplicate trip IDs, physically impossible
records (negative fares, teleporting trips), 15 different spellings of four
payment methods, and missing zones/tips. Build the deterministic cleaning
pipeline that turns it into an analysis-ready table.

Implement `solve() -> pl.DataFrame`.

## Required pipeline (in any order that produces the spec'd result)

1. **Parse timestamps** — `pickup_datetime`, `dropoff_datetime` → `Datetime`
   using format `"%Y-%m-%d %H:%M:%S"`.
2. **Derive** `trip_duration_min` = minutes between dropoff and pickup, and
   `implied_speed_kmh` = `distance_km / (trip_duration_min / 60)`.
3. **Normalise `payment_type`** to exactly four canonical values. Match
   case-insensitively on substrings:
   - contains `grab` → `"Grab"`
   - contains `nets` → `"NETS"`
   - contains `cash` → `"Cash"`
   - contains `card`, `visa`, `mastercard`, or `credit` → `"Card"`
4. **Impute** — `tip_sgd` null → `0.0`; `pickup_zone` / `dropoff_zone` null →
   `"Unknown"`.
5. **Drop physically impossible rows** — keep a row only if ALL hold:
   - `fare_sgd > 0`
   - `0 < distance_km <= 100`
   - `passengers >= 1`
   - `0 < trip_duration_min <= 180`
   - `2 <= implied_speed_kmh <= 120`
6. **Deduplicate** by `trip_id`, keeping the row with the **highest `fare_sgd`**
   (tie-break: latest `dropoff_datetime`). Exactly one row per `trip_id`.
7. **Derive** `fare_per_km` = `fare_sgd / distance_km`, and `is_airport`
   (Boolean) = `True` when `pickup_zone` **or** `dropoff_zone` is
   `"Changi Airport"`.
8. **Return** a DataFrame with these **16 columns in this exact order**, sorted
   ascending by `pickup_datetime`:

   ```
   trip_id, pickup_datetime, dropoff_datetime, pickup_zone, dropoff_zone,
   distance_km, fare_sgd, tip_sgd, payment_type, passengers,
   pickup_latitude, pickup_longitude, trip_duration_min, implied_speed_kmh,
   fare_per_km, is_airport
   ```

## Visible sanity checks

After a correct implementation:

- `result.shape == (44596, 16)`
- `sorted(result["payment_type"].unique()) == ["Card", "Cash", "Grab", "NETS"]`
- every `implied_speed_kmh` lies in `[2, 120]`
- `result["trip_id"].n_unique() == result.height` (no duplicates)
- about 2,400 airport trips

## Grading (10 automated checks, all must pass)

return type · exact 16-column schema · datetime dtypes · payment normalised to
the 4 labels · no nulls in key columns · plausibility invariants (no impossible
row survives) · no duplicate `trip_id` · row count matches the independently
re-derived ground truth · derived columns correct · sorted by pickup.

## Rules

- **Polars only** — no pandas.
- Load via `shared.MLFPDataLoader` (works in VS Code and Colab).
- Cleaning must be **deterministic** — no random sampling.
