#!/usr/bin/env python
"""Runner for Task 4 tests"""
import sys
import os

sys.path.insert(0, '.')
os.chdir('./modules/mlfp01/assessment/task_4')

from starter import solve
import polars as pl


def run_tests():
    """Run all tests and report results."""
    
    def test(name, fn):
        try:
            fn()
            print(f"✓ {name}")
        except AssertionError as e:
            print(f"✗ {name}: {e}")
        except Exception as e:
            print(f"✗ {name}: {type(e).__name__}: {e}")

    print("=" * 70)
    print("TASK 4 PROFILE, CLEAN & DATAEXPLORER TEST SUITE")
    print("=" * 70)

    # Run solve once for all tests
    result = solve()
    df = result["cleaned"]

    # ========== Schema Tests ==========
    print("\n[Schema Tests]")
    
    def test_return_dict():
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "cleaned" in result, "Missing 'cleaned' key"
        assert "raw_alert_count" in result, "Missing 'raw_alert_count' key"
        assert "clean_alert_count" in result, "Missing 'clean_alert_count' key"
    test("Return dict has required keys", test_return_dict)

    def test_cleaned_is_dataframe():
        assert isinstance(df, pl.DataFrame), f"Expected DataFrame, got {type(df)}"
    test("cleaned is DataFrame", test_cleaned_is_dataframe)

    def test_output_shape():
        assert df.shape == (101, 8), f"Expected (101, 8), got {df.shape}"
    test("Output shape (101, 8)", test_output_shape)

    def test_column_names():
        expected = ["period_year", "period_quarter", "gdp_growth_pct", "unemployment_rate",
                    "inflation_rate", "trade_balance_sgd_bn", "property_price_index", "tourist_arrivals"]
        actual = df.columns
        assert actual == expected, f"Expected {expected}, got {actual}"
    test("Column names and order", test_column_names)

    def test_column_dtypes():
        expected_dtypes = {
            "period_year": pl.Int64,
            "period_quarter": pl.Int64,
            "gdp_growth_pct": pl.Float64,
            "unemployment_rate": pl.Float64,
            "inflation_rate": pl.Float64,
            "trade_balance_sgd_bn": pl.Float64,
            "property_price_index": pl.Float64,
            "tourist_arrivals": pl.Int64,
        }
        for col, expected_dtype in expected_dtypes.items():
            actual = df[col].dtype
            assert actual == expected_dtype, f"Column {col}: expected {expected_dtype}, got {actual}"
    test("Column data types correct", test_column_dtypes)

    # ========== Null Handling Tests ==========
    print("\n[Null Handling Tests]")

    def test_no_nulls_in_imputed():
        for col in ["inflation_rate", "trade_balance_sgd_bn", "tourist_arrivals"]:
            null_count = df[col].null_count()
            assert null_count == 0, f"Column {col} has {null_count} nulls, expected 0"
    test("No nulls in imputed/converted columns", test_no_nulls_in_imputed)

    def test_imputation_values():
        # Check that inflation_rate and trade_balance were imputed with reasonable values
        inflation = df["inflation_rate"]
        trade = df["trade_balance_sgd_bn"]
        # Both should have numeric values
        assert inflation.min() > -10, f"inflation_rate min suspiciously low: {inflation.min()}"
        assert trade.min() > -100, f"trade_balance min suspiciously low: {trade.min()}"
    test("Imputed values in reasonable range", test_imputation_values)

    # ========== Feature Correctness Tests ==========
    print("\n[Feature Correctness Tests]")

    def test_period_year_range():
        min_year = int(df["period_year"].min())
        max_year = int(df["period_year"].max())
        assert min_year >= 2000, f"Min year {min_year} too low"
        assert max_year <= 2025, f"Max year {max_year} too high"
    test("period_year in reasonable range", test_period_year_range)

    def test_period_quarter_range():
        min_q = int(df["period_quarter"].min())
        max_q = int(df["period_quarter"].max())
        assert min_q == 1, f"Min quarter: expected 1, got {min_q}"
        assert max_q == 4, f"Max quarter: expected 4, got {max_q}"
    test("period_quarter range 1-4", test_period_quarter_range)

    def test_tourist_arrivals_int64():
        assert df["tourist_arrivals"].dtype == pl.Int64, f"Expected Int64, got {df['tourist_arrivals'].dtype}"
        # Check values are reasonable (non-negative)
        min_arrivals = int(df["tourist_arrivals"].min())
        assert min_arrivals >= 0, f"Min tourist arrivals should be >= 0, got {min_arrivals}"
    test("tourist_arrivals is Int64 with non-negative values", test_tourist_arrivals_int64)

    def test_gdp_growth_in_range():
        gdp = df["gdp_growth_pct"]
        min_gdp = float(gdp.min())
        max_gdp = float(gdp.max())
        assert min_gdp > -20, f"GDP min too low: {min_gdp}"
        assert max_gdp < 20, f"GDP max too high: {max_gdp}"
    test("gdp_growth_pct in reasonable range", test_gdp_growth_in_range)

    def test_unemployment_in_range():
        unemp = df["unemployment_rate"]
        min_unemp = float(unemp.min())
        max_unemp = float(unemp.max())
        assert min_unemp >= 0, f"Unemployment min should be >= 0, got {min_unemp}"
        assert max_unemp <= 15, f"Unemployment max should be <= 15, got {max_unemp}"
    test("unemployment_rate in reasonable range", test_unemployment_in_range)

    # ========== Sorting & Uniqueness Tests ==========
    print("\n[Sorting & Uniqueness Tests]")

    def test_sorted_by_year_quarter():
        sorted_df = df.sort(["period_year", "period_quarter"])
        assert df.to_dicts() == sorted_df.to_dicts(), "DataFrame not sorted by [period_year, period_quarter]"
    test("Sorted by [period_year, period_quarter]", test_sorted_by_year_quarter)

    def test_year_quarter_unique():
        n_rows = df.height
        n_unique = df.group_by(["period_year", "period_quarter"]).agg(pl.len()).height
        # Note: Data has a duplicate (2019, Q2) which is preserved as-is
        # (problem requires 101 rows total from quarterly slice)
        assert n_rows == 101, f"Expected 101 rows (including duplicate), got {n_rows}"
        assert n_unique == 100, f"Expected 100 unique (year, quarter) pairs, got {n_unique}"
    test("(period_year, period_quarter) has expected duplicate", test_year_quarter_unique)

    # ========== DataExplorer Alert Tests ==========
    print("\n[DataExplorer Alert Tests]")

    def test_raw_alert_count_type():
        assert isinstance(result["raw_alert_count"], int), f"raw_alert_count should be int, got {type(result['raw_alert_count'])}"
    test("raw_alert_count is int", test_raw_alert_count_type)

    def test_clean_alert_count_type():
        assert isinstance(result["clean_alert_count"], int), f"clean_alert_count should be int, got {type(result['clean_alert_count'])}"
    test("clean_alert_count is int", test_clean_alert_count_type)

    def test_cleaning_reduced_alerts():
        raw = result["raw_alert_count"]
        clean = result["clean_alert_count"]
        assert clean < raw, f"Cleaning should reduce alerts: raw={raw}, clean={clean}"
    test("Cleaning reduced alert count", test_cleaning_reduced_alerts)

    def test_alert_counts_positive():
        raw = result["raw_alert_count"]
        clean = result["clean_alert_count"]
        assert raw > 0, f"raw_alert_count should be > 0, got {raw}"
        assert clean >= 0, f"clean_alert_count should be >= 0, got {clean}"
    test("Alert counts are non-negative", test_alert_counts_positive)

    # ========== Integration Tests ==========
    print("\n[Integration Tests]")

    def test_deterministic():
        result2 = solve()
        df2 = result2["cleaned"]
        assert df.to_dicts() == df2.to_dicts(), "solve() not deterministic"
    test("Deterministic (consistent) output", test_deterministic)

    print("\n" + "=" * 70)
    print("TEST SUITE COMPLETE")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Cleaned DataFrame shape: {df.shape}")
    print(f"  Raw alert count: {result['raw_alert_count']}")
    print(f"  Clean alert count: {result['clean_alert_count']}")
    print(f"  Alert reduction: {result['raw_alert_count'] - result['clean_alert_count']} alerts removed")


if __name__ == "__main__":
    run_tests()
