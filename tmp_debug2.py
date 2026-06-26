import sys
from pathlib import Path
repo = Path('.').resolve()
sys.path.insert(0, str(repo))
from shared import MLFPDataLoader
import polars as pl

loader = MLFPDataLoader()
df = loader.load('mlfp01', 'sg_taxi_trips.parquet')
print('raw shape', df.shape)
print('raw columns', df.columns)
print('raw dtypes', [str(dt) for dt in df.dtypes])
print('raw first rows:')
print(df.head(5))

# parse datetimes
parsed = df.with_columns([
    pl.col('pickup_datetime').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S').alias('pickup_datetime'),
    pl.col('dropoff_datetime').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S').alias('dropoff_datetime'),
    pl.col('tip_sgd').fill_null(0.0).alias('tip_sgd'),
    pl.col('pickup_zone').fill_null('Unknown').alias('pickup_zone'),
    pl.col('dropoff_zone').fill_null('Unknown').alias('dropoff_zone'),
])
print('parsed dtypes', [str(dt) for dt in parsed.dtypes])
print(parsed.select(['pickup_datetime','dropoff_datetime']).head(5))

parsed = parsed.with_columns([
    ((pl.col('dropoff_datetime').cast(pl.Int64) - pl.col('pickup_datetime').cast(pl.Int64)) / 1e9 / 60.0).alias('trip_duration_min'),
    (pl.col('distance_km') * 60.0 / pl.col('trip_duration_min')).alias('implied_speed_kmh'),
])
print('derived first rows:')
print(parsed.select(['distance_km','trip_duration_min','implied_speed_kmh']).head(10))
print('derived stats:')
print(parsed.select([
    pl.col('trip_duration_min').min().alias('min_dur'),
    pl.col('trip_duration_min').max().alias('max_dur'),
    pl.col('implied_speed_kmh').min().alias('min_speed'),
    pl.col('implied_speed_kmh').max().alias('max_speed'),
]).collect())

# counts by filters
filters = {
    'fare_sgd > 0': parsed.filter(pl.col('fare_sgd') > 0).height,
    'distance_km > 0': parsed.filter(pl.col('distance_km') > 0).height,
    'distance_km <= 100': parsed.filter(pl.col('distance_km') <= 100).height,
    'passengers >= 1': parsed.filter(pl.col('passengers') >= 1).height,
    'trip_duration_min > 0': parsed.filter(pl.col('trip_duration_min') > 0).height,
    'trip_duration_min <= 180': parsed.filter(pl.col('trip_duration_min') <= 180).height,
    'speed >= 2': parsed.filter(pl.col('implied_speed_kmh') >= 2).height,
    'speed <= 120': parsed.filter(pl.col('implied_speed_kmh') <= 120).height,
}
print('filter counts')
for k,v in filters.items():
    print(k, v)

print('rows remaining after all filters:', parsed.filter(
    (pl.col('fare_sgd') > 0)
    & (pl.col('distance_km') > 0)
    & (pl.col('distance_km') <= 100)
    & (pl.col('passengers') >= 1)
    & (pl.col('trip_duration_min') > 0)
    & (pl.col('trip_duration_min') <= 180)
    & (pl.col('implied_speed_kmh') >= 2)
    & (pl.col('implied_speed_kmh') <= 120)
).height)
