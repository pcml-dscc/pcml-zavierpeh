import sys
from pathlib import Path
repo = Path('.').resolve()
sys.path.insert(0, str(repo))
from shared import MLFPDataLoader
import polars as pl

loader = MLFPDataLoader()
df = loader.load('mlfp01', 'sg_taxi_trips.parquet')
print('raw shape', df.shape)
print('columns', df.columns)
print('dtypes', [str(dt) for dt in df.dtypes])
print(df.head(5))
print('pickup_datetime sample type', df['pickup_datetime'].dtype)
print('first pickup_datetime sample', df['pickup_datetime'].head(5).to_list())
print('payment_type sample unique', sorted(df['payment_type'].unique().to_list()))
print('null counts')
for c in ['pickup_datetime','dropoff_datetime','payment_type','tip_sgd','pickup_zone','dropoff_zone']:
    print(c, df[c].null_count())
