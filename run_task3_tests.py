#!/usr/bin/env python
"""Runner for Task 3 tests"""
import sys
import os

sys.path.insert(0, '.')
os.chdir('./modules/mlfp01/assessment/task_3')

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
    print("TASK 3 WINDOW FUNCTIONS & PRICE TRENDS TEST SUITE")
    print("=" * 70)

    # ========== Schema Tests ==========
    print("\n[Schema Tests]")
    
    def test_output_shape():
        df = solve()
        assert df.shape == (270, 7), f"Expected (270, 7), got {df.shape}"
    test("Output shape (270, 7)", test_output_shape)

    def test_column_names():
        df = solve()
        expected = ["town", "sale_year", "n_sales", "median_price", "yoy_pct", "rolling_3yr_avg", "price_rank_in_year"]
        actual = df.columns
        assert actual == expected, f"Expected {expected}, got {actual}"
    test("Column names and order", test_column_names)

    def test_column_dtypes():
        df = solve()
        expected_dtypes = {
            "town": pl.Utf8,
            "sale_year": pl.Int64,
            "n_sales": pl.UInt32,
            "median_price": pl.Float64,
            "yoy_pct": pl.Float64,
            "rolling_3yr_avg": pl.Float64,
            "price_rank_in_year": pl.UInt32,
        }
        for col, dtype in expected_dtypes.items():
            actual = df[col].dtype
            # Allow some type variation (Int64 vs UInt32 for counts)
            if col in ["n_sales", "price_rank_in_year"]:
                assert actual in [pl.UInt32, pl.Int64, pl.Int32], f"Column {col}: expected integer type, got {actual}"
            else:
                assert actual == dtype, f"Column {col}: expected {dtype}, got {actual}"
    test("Column data types", test_column_dtypes)

    # ========== Null Handling Tests ==========
    print("\n[Null Handling Tests]")

    def test_yoy_pct_nulls():
        df = solve()
        null_count = df["yoy_pct"].null_count()
        assert null_count == 27, f"Expected 27 null yoy_pct (one per town), got {null_count}"
    test("yoy_pct nulls (27, one per town)", test_yoy_pct_nulls)

    def test_no_other_nulls():
        df = solve()
        for col in ["town", "sale_year", "n_sales", "median_price", "rolling_3yr_avg", "price_rank_in_year"]:
            null_count = df[col].null_count()
            assert null_count == 0, f"Column {col} has {null_count} nulls, expected 0"
    test("No nulls in other columns", test_no_other_nulls)

    # ========== Feature Correctness Tests ==========
    print("\n[Feature Correctness Tests]")

    def test_sale_year_range():
        df = solve()
        min_year = int(df["sale_year"].min())
        max_year = int(df["sale_year"].max())
        assert min_year == 2015, f"Min sale_year: expected 2015, got {min_year}"
        assert max_year == 2024, f"Max sale_year: expected 2024, got {max_year}"
    test("sale_year range (2015-2024)", test_sale_year_range)

    def test_n_sales_positive():
        df = solve()
        min_sales = int(df["n_sales"].min())
        assert min_sales > 0, f"Min n_sales should be > 0, got {min_sales}"
    test("n_sales all positive", test_n_sales_positive)

    def test_median_price_positive():
        df = solve()
        min_price = float(df["median_price"].min())
        assert min_price > 0, f"Min median_price should be > 0, got {min_price}"
    test("median_price all positive", test_median_price_positive)

    def test_price_rank_in_year_range():
        df = solve()
        min_rank = int(df["price_rank_in_year"].min())
        max_rank = int(df["price_rank_in_year"].max())
        assert min_rank == 1, f"Min rank: expected 1, got {min_rank}"
        assert max_rank == 27, f"Max rank: expected 27, got {max_rank}"
    test("price_rank_in_year range (1-27)", test_price_rank_in_year_range)

    def test_yoy_pct_non_null_range():
        df = solve()
        yoy_non_null = df.filter(pl.col("yoy_pct").is_not_null())["yoy_pct"]
        assert len(yoy_non_null) > 0, "No non-null yoy_pct values found"
        # Check reasonable bounds: -50% to +50% YoY changes
        assert float(yoy_non_null.min()) > -100, f"Minimum yoy_pct too low: {yoy_non_null.min()}"
        assert float(yoy_non_null.max()) < 100, f"Maximum yoy_pct too high: {yoy_non_null.max()}"
    test("yoy_pct reasonable range (-100% to +100%)", test_yoy_pct_non_null_range)

    def test_rolling_avg_bounds():
        df = solve()
        # Rolling average should be between min and max of median_price
        rolling = df["rolling_3yr_avg"]
        prices = df["median_price"]
        min_rolling = float(rolling.min())
        max_rolling = float(rolling.max())
        min_price = float(prices.min())
        max_price = float(prices.max())
        # Sanity: rolling avg should roughly be in the price range
        assert min_rolling >= min_price * 0.9, f"Min rolling avg {min_rolling} suspiciously low vs min price {min_price}"
        assert max_rolling <= max_price * 1.1, f"Max rolling avg {max_rolling} suspiciously high vs max price {max_price}"
    test("rolling_3yr_avg within reasonable bounds", test_rolling_avg_bounds)

    # ========== Sorting & Uniqueness Tests ==========
    print("\n[Sorting & Uniqueness Tests]")

    def test_sorted_by_town_year():
        df = solve()
        sorted_df = df.sort(["town", "sale_year"])
        assert df.to_dicts() == sorted_df.to_dicts(), "DataFrame not sorted by [town, sale_year]"
    test("Sorted by [town, sale_year]", test_sorted_by_town_year)

    def test_town_year_unique():
        df = solve()
        n_rows = df.height
        n_unique = df.group_by(["town", "sale_year"]).agg(pl.len()).height
        assert n_rows == n_unique, f"Duplicate (town, sale_year) found: {n_rows} rows but {n_unique} unique"
    test("(town, sale_year) is unique", test_town_year_unique)

    def test_27_unique_towns():
        df = solve()
        n_towns = df["town"].n_unique()
        assert n_towns == 27, f"Expected 27 unique towns, got {n_towns}"
    test("27 unique towns", test_27_unique_towns)

    def test_10_unique_years():
        df = solve()
        n_years = df["sale_year"].n_unique()
        assert n_years == 10, f"Expected 10 unique years, got {n_years}"
    test("10 unique years (2015-2024)", test_10_unique_years)

    # ========== Window Function Logic Tests ==========
    print("\n[Window Function Logic Tests]")

    def test_yoy_pct_calculation():
        df = solve()
        # Verify YoY calculation: yoy_pct = 100 * (median_price - prev_median) / prev_median
        # Pick a town and check one value
        town_df = df.filter(pl.col("town") == df["town"][0]).sort("sale_year")
        if town_df.height >= 2:
            # Second year's YoY should be calculable
            curr_price = float(town_df["median_price"][1])
            prev_price = float(town_df["median_price"][0])
            expected_yoy = 100 * (curr_price - prev_price) / prev_price
            actual_yoy = float(town_df["yoy_pct"][1])
            rel_error = abs(actual_yoy - expected_yoy) / abs(expected_yoy) if expected_yoy != 0 else 0
            assert rel_error < 0.01, f"YoY mismatch: expected {expected_yoy:.2f}, got {actual_yoy:.2f}"
    test("yoy_pct calculation accuracy", test_yoy_pct_calculation)

    def test_rolling_avg_calculation():
        df = solve()
        # For first year rolling avg should equal median_price
        town_df = df.filter(pl.col("town") == df["town"][0]).sort("sale_year")
        if town_df.height >= 1:
            first_rolling = float(town_df["rolling_3yr_avg"][0])
            first_price = float(town_df["median_price"][0])
            assert abs(first_rolling - first_price) < 0.01, f"First year rolling avg should equal median_price"
    test("rolling_3yr_avg first-year calculation", test_rolling_avg_calculation)

    def test_price_rank_within_year():
        df = solve()
        # For each year, check that rank distribution is correct
        for year in [2015, 2024]:
            year_df = df.filter(pl.col("sale_year") == year)
            ranks = set(year_df["price_rank_in_year"].to_list())
            # For ties, Polars rank(method="min") skips ranks
            assert 1 in ranks, f"Year {year}: rank 1 not found"
            min_rank = min(ranks)
            max_rank = max(ranks)
            assert min_rank == 1, f"Year {year}: min rank should be 1, got {min_rank}"
            assert max_rank <= 27, f"Year {year}: max rank should be <=27, got {max_rank}"
    test("price_rank_in_year within year grouping", test_price_rank_within_year)

    # ========== Integration Tests ==========
    print("\n[Integration Tests]")

    def test_deterministic():
        df1 = solve()
        df2 = solve()
        assert df1.to_dicts() == df2.to_dicts(), "solve() not deterministic"
    test("Deterministic (consistent) output", test_deterministic)

    def test_return_type():
        result = solve()
        assert isinstance(result, pl.DataFrame), f"Expected DataFrame, got {type(result)}"
    test("Return type is DataFrame", test_return_type)

    print("\n" + "=" * 70)
    print("TEST SUITE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
