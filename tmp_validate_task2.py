import sys
from pathlib import Path
repo = Path('.').resolve()
sys.path.insert(0, str(repo))
from modules.mlfp01.assessment.task_2.starter import solve
import polars as pl

df = solve()
print('shape', df.shape)
print('columns', df.columns)
print('null storey_midpoint', df['storey_midpoint'].null_count())
print('null remaining_lease_years', df['remaining_lease_years'].null_count())
print('flat_type_rooms uniques', sorted(df['flat_type_rooms'].unique().to_list()))
print('first rows')
print(df.head(5))
print('storey parse failures', df.filter(pl.col('storey_midpoint').is_null()).height)
print('remaining lease parse failures', df.filter(pl.col('remaining_lease_years').is_null()).height)
