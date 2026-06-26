import sys
from pathlib import Path
repo = Path('.').resolve()
sys.path.insert(0, str(repo))
from modules.mlfp01.assessment.task_1.starter import solve
import polars as pl

result = solve()
print('shape', result.shape)
print('columns', result.columns)
print('payment_type', sorted(result['payment_type'].unique().to_list()))
print('trip_id_unique', result['trip_id'].n_unique())
print('min_speed', result['implied_speed_kmh'].min())
print('max_speed', result['implied_speed_kmh'].max())
print('airport', result.filter(pl.col('is_airport')).height)
assert result.shape == (44596, 16), 'Unexpected shape'
assert sorted(result['payment_type'].unique().to_list()) == ['Card', 'Cash', 'Grab', 'NETS'], 'Unexpected payment labels'
assert result['implied_speed_kmh'].min() >= 2, 'implied_speed min < 2'
assert result['implied_speed_kmh'].max() <= 120, 'implied_speed max > 120'
assert result['trip_id'].n_unique() == result.height, 'duplicate trip_id detected'
print('All checks passed')
