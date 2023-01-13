# %%
import os
import pandas as pd
import numpy as np

RESULTS_ROOT = 'results-2023-01-07'

metrics = ['FOR', 'FDR', 'AUC']

metric_dfs = {}
for metric in metrics:
    metric_df = pd.read_csv(
        os.path.join(
            RESULTS_ROOT,
            'results_BINARYdata_fdr_0.10',
            f'BINARYdata-all-{metric}-mean.csv'
        ),
        header=[0, 1],
        index_col=[0]
    )
    metric_dfs[metric] = metric_df

for metric in metrics:
    metric_df = metric_dfs[metric]
    for_ctl_methods = [
        (method, cutoff) for method, cutoff in metric_df.columns \
        if 'Multisplit+FOR-CTL' in cutoff
    ]
    metric_df = metric_df[for_ctl_methods]
    metric_dfs[metric] = metric_df.iloc[:-1, :-3]

metric_dfs['FOR'] = metric_dfs['FOR'] \
    .sort_values([('IForest', 'Multisplit+FOR-CTL')])

# (metric_dfs['FOR'] < 0.2).sum(axis=1).reset_index(drop=False) \
#     .sort_values(0)
# (metric_dfs['FOR'] < 0.2).sum(axis=0)
metric_dfs['FOR']

# %%
dataset_order = metric_dfs['FOR'].index
dataset_order

# %%
for metric in metrics:
    metric_dfs[metric] = metric_dfs[metric] \
        .loc[dataset_order]

# %%
metric_dfs['AUC']

# %%
