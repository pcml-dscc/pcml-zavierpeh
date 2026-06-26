import sys
from pathlib import Path
repo = Path('.').resolve()
sys.path.insert(0, str(repo))
from shared import MLFPDataLoader
import polars as pl

loader = MLFPDataLoader()
df = loader.load('mlfp01', 'sg_taxi_trips.parquet')
print('raw shape', df.shape)

# parse datetimes and impute
step1 = df.with_columns([
    pl.col('pickup_datetime').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S').alias('pickup_datetime'),
    pl.col('dropoff_datetime').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S').alias('dropoff_datetime'),
    pl.lit('Grab').alias('tmp'),
]).drop(['tmp'])
print('step1 shape', step1.shape)
print(step1.select(['pickup_datetime','dropoff_datetime']).head(5))

step2 = step1.with_columns([
    pl.when(pl.col('payment_type').str.to_lowercase().str.contains('grab')).then(pl.lit('Grab'))
      .when(pl.col('payment_type').str.to_lowercase().str.contains('nets')).then(pl.lit('NETS'))
      .when(pl.col('payment_type').str.to_lowercase().str.contains('cash')).then(pl.lit('Cash'))
      .when(
           pl.col('payment_type').str.to_lowercase().str.contains('card')
          | pl.col('payment_type').str.to_lowercase().str.contains('visa')
          | pl.col('payment_type').str.to_lowercase().str.contains('mastercard')
          | pl.col('payment_type').str.to_lowercase().str.contains('credit')
        ).then(pl.lit('Card'))
      .otherwise(pl.col('payment_type')).alias('payment_type'),
    pl.col('tip_sgd').fill_null(0.0).alias('tip_sgd'),
    pl.col('pickup_zone').fill_null('Unknown').alias('pickup_zone'),
    pl.col('dropoff_zone').fill_null('Unknown').alias('dropoff_zone'),
])
print('step2 payment_type uniques', sorted(step2['payment_type'].unique().to_list()))
print('step2 null counts', {c: step2[c].null_count() for c in ['payment_type','tip_sgd','pickup_zone','dropoff_zone']})

step3 = step2.with_columns([
    ((pl.col('dropoff_datetime').cast(pl.Int64) - pl.col('pickup_datetime').cast(pl.Int64)) / 1e9 / 60.0).alias('trip_duration_min')
])
print('step3 trip_duration_min stats', step3.select([
    pl.col('trip_duration_min').min().alias('min_dur'),
    pl.col('trip_duration_min').max().alias('max_dur'),
]).to_dict())
step4 = step3.with_columns([(pl.col('distance_km') * 60.0 / pl.col('trip_duration_min')).alias('implied_speed_kmh')])
print('step4 implied_speed_kmh stats', step4.select([
    pl.col('implied_speed_kmh').min().alias('min_speed'),
    pl.col('implied_speed_kmh').max().alias('max_speed'),
]).to_dict())

# counts on filters
for name, expr in [
    ('fare_sgd > 0', pl.col('fare_sgd') > 0),
    ('distance_km > 0', pl.col('distance_km') > 0),
    ('distance_km <= 100', pl.col('distance_km') <= 100),
    ('passengers >= 1', pl.col('passengers') >= 1),
    ('duration > 0', pl.col('trip_duration_min') > 0),
    ('duration <= 180', pl.col('trip_duration_min') <= 180),
    ('speed >= 2', pl.col('implied_speed_kmh') >= 2),
    ('speed <= 120', pl.col('implied_speed_kmh') <= 120),
]:
    print(name, step4.filter(expr).height)

filtered = step4.filter(
    (pl.col('fare_sgd') > 0)
    & (pl.col('distance_km') > 0)
    & (pl.col('distance_km') <= 100)
    & (pl.col('passengers') >= 1)
    & (pl.col('trip_duration_min') > 0)
    & (pl.col('trip_duration_min') <= 180)
    & (pl.col('implied_speed_kmh') >= 2)
    & (pl.col('implied_speed_kmh') <= 120)
)
print('after all filters', filtered.height)
print(filtered.select(['distance_km','trip_duration_min','implied_speed_kmh']).head(20))
