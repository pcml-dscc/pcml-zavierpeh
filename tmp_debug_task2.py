import sys
from pathlib import Path
repo = Path('.').resolve()
sys.path.insert(0, str(repo))
from shared import MLFPDataLoader
import polars as pl

loader = MLFPDataLoader()
df = loader.load('mlfp01', 'hdb_resale.parquet')
print('shape', df.shape)
print('columns', df.columns)
print('dtypes', [str(dt) for dt in df.dtypes])
print(df.head(10))
print('storey_range uniqs', df['storey_range'].unique().sort().to_list()[:40])
print('remaining_lease uniqs', df['remaining_lease'].unique().sort().to_list()[:40])
print('flat_type uniqs', df['flat_type'].unique().sort().to_list())
print('month sample', df['month'].unique().sort().to_list()[:20])
PY