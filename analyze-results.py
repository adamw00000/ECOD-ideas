# %%
import os
import pandas as pd
import numpy as np

RESULTS_ROOT = 'results-2023-01-07'

pd.set_option('display.max_colwidth', 1000)
metrics = ['FOR']

alpha = 0.1

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
    sem_df = pd.read_csv(
        os.path.join(
            RESULTS_ROOT,
            'results_BINARYdata_fdr_0.10',
            f'BINARYdata-all-{metric}-sem.csv'
        ),
        header=[0, 1],
        index_col=[0]
    )
        
    metric_dfs[metric] = '$' \
        + metric_df.applymap('{:.3f}'.format) \
        + ' \pm ' \
        + sem_df.applymap('{:.3f}'.format) \
        + '\\,' \
        + np.where(metric_df <= 2 * alpha, '\\textcolor{black}{\\checkmark}', '\\phantom{\\checkmark}') \
        + np.where(metric_df <= alpha, '\\textcolor{magenta}{\\checkmark}', '\\phantom{\\checkmark}')\
        + '$'

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

no_cutoff_row_cols = [method for method, c in metric_dfs['FOR'].columns]
metric_dfs['FOR'].columns = no_cutoff_row_cols

metric_dfs['FOR'] = metric_dfs['FOR'][['IForest', 'A^3', 'Mahalanobis', 'ECODv2', 'ECODv2+PCA1.0']] \
    .rename(columns={'ECODv2': 'ECOD', 'ECODv2+PCA1.0': 'ECOD + PCA', 'A^3': '$A^3$'})

metric_dfs['FOR'].index = '\\textit{' \
    + metric_dfs['FOR'].index.str.replace('\(csv\) ', '') \
    + '}'
metric_dfs['FOR'].columns = ['\\textbf{' + c + '}' for c in metric_dfs['FOR'].columns]
metric_dfs['FOR'].columns.name = '\\textbf{Dataset}'
# (metric_dfs['FOR'] < 0.2).sum(axis=1).reset_index(drop=False) \
#     .sort_values(0)
# (metric_dfs['FOR'] < 0.2).sum(axis=0)
with open(os.path.join('latex', 'for-table.tex'), 'w+') as f:
    f.write(
        metric_dfs['FOR'] \
            .to_latex(
                escape=False, 
                # bold_rows=True,
                column_format='l|ccccc',
            )
    )

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
