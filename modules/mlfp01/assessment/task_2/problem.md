# MLFP01 — Task 2: HDB Feature Engineering

**Weight**: 25 marks · **Difficulty**: Hard · **Dataset**: `data/mlfp01/hdb_resale.parquet` (50,150 rows, 11 columns)

## Scenario

Raw HDB resale records carry features locked inside messy strings: storey
bands with OCR errors, lease durations in two incompatible formats, and
categorical flat types. Engineer a clean, numeric, model-ready feature table.
**No rows are dropped** — this is feature engineering, not filtering.

Implement `solve() -> pl.DataFrame`.

## Required features

1. **`sale_year`** (Int) — the year from the `"YYYY-MM"` `month` string.
2. **`storey_midpoint`** (Float) — parse `storey_range` of the form
   `"LO TO HI"` and average the two bounds. **Gotcha:** some digits were OCR'd
   as the capital letter `O` (e.g. `"4O TO 42"`, `"1O TO 12"`). Fix the numeric
   tokens — but note the delimiter `" TO "` legitimately contains an `O`, so a
   blanket replace will corrupt it. Split first, fix the numbers second.
3. **`flat_age_years`** (Int) — `sale_year - lease_commence_date`.
4. **`price_per_sqm`** (Float) — `resale_price / floor_area_sqm`.
5. **`flat_type_rooms`** (Int) — ordinal encoding:
   `2 ROOM→2, 3 ROOM→3, 4 ROOM→4, 5 ROOM→5, EXECUTIVE→6, MULTI-GENERATION→7`.
6. **`remaining_lease_years`** (Float) — parse the dual-format `remaining_lease`:
   - `"71 years 11 months"` → `71 + 11/12`
   - bare `"92"` → `92.0`
   - **null** (1,474 rows) → impute as `99 - flat_age_years` (HDB flats are
     sold on a statutory 99-year lease).

## Output schema (10 columns, this exact order), sorted by `[sale_year, town]`

```
town, flat_type, flat_type_rooms, sale_year, storey_midpoint,
floor_area_sqm, flat_age_years, remaining_lease_years, resale_price, price_per_sqm
```

## Visible sanity checks

- `result.shape == (50150, 10)` (no rows dropped)
- `result["storey_midpoint"].null_count() == 0` (every band parsed)
- `result["remaining_lease_years"].null_count() == 0` (every lease parsed/imputed)
- `sorted(result["flat_type_rooms"].unique()) == [2, 3, 4, 5, 6, 7]`

## Grading (10 automated checks, all must pass)

return type · exact 10-column schema · row count 50,150 · no nulls in derived
columns · `storey_midpoint` matches reference (OCR fix correct) ·
`remaining_lease_years` matches reference (both formats + imputation) ·
`flat_type_rooms` correct · `flat_age_years` correct · `price_per_sqm` correct ·
sorted by `[sale_year, town]`.

## Rules

- **Polars only** — no pandas. Load via `shared.MLFPDataLoader`. Deterministic.
