# %%
import os
import pandas as pd
import numpy as np

RESULTS_ROOT = 'results-2023-01-04'

for_df = pd.read_csv(
    os.path.join(
        RESULTS_ROOT,
        'results_BINARYdata_fdr_0.10',
        'BINARYdata-all-FOR-mean.csv'
    ),
    header=[0, 1],
    index_col=[0]
)
for_df

# %%
for_ctl_methods = [
    (method, cutoff) for method, cutoff in for_df.columns \
    if 'Multisplit+FOR-CTL' in cutoff
]
for_df = for_df[for_ctl_methods]
for_df = for_df.iloc[:-1]
for_df.sort_values([('IForest', 'Multisplit+FOR-CTL')])

# %%
