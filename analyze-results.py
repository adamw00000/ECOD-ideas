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
    .rename(columns={'ECODv2': 'ECOD', 'ECODv2+PCA1.0': 'ECOD + PCA', 'A^3': '$\\bm{A^3}$'})

metric_dfs['FOR'].index = '\\textit{' \
    + metric_dfs['FOR'].index.str.replace('\(csv\) ', '') \
        .str.replace('authentication', 'auth') \
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
skew_df = pd.DataFrame()

for dataset in os.listdir(os.path.join('.', 'results_BINARYdata_fdr_0.10')):
    dataset_dir = os.path.join('.', 'results_BINARYdata_fdr_0.10', dataset)
    if not os.path.isdir(dataset_dir) or 'global' in dataset_dir:
        continue

    if os.path.exists(os.path.join(dataset_dir, 'pval_metrics.csv')):
        df = pd.read_csv(os.path.join(dataset_dir, 'pval_metrics.csv'))
        df = df[df.Metric == 'Skewness']
        df = df.assign(Dataset=dataset)

        skew_df = pd.concat([
            skew_df,
            df.pivot_table(values='Value', index='Dataset', columns=['Type'])
        ])

skew_df['SkewDiff'] = skew_df.Outlier - skew_df.Inlier
display(skew_df)

# %%
raw_for_df = pd.read_csv(
    os.path.join(
        RESULTS_ROOT,
        'results_BINARYdata_fdr_0.10',
        f'BINARYdata-all-FOR-mean.csv'
    ),
    header=[0, 1],
    index_col=[0]
).iloc[:-1][('IForest', 'Multisplit+FOR-CTL')]
raw_for_df

# %%
raw_df = pd.read_csv(
    os.path.join(
        RESULTS_ROOT,
        'results_BINARYdata_fdr_0.10',
        f'BINARYdata-raw-results.csv'
    ),
    # header=[0, 1],
    # index_col=[0]
)
raw_for_df_v2 = raw_df[(raw_df.Method == 'IForest') & (raw_df.Cutoff == 'Multisplit+FOR-CTL')] \
    .groupby('Dataset') \
    .mean() \
    .FOR

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
plt.figure(figsize=(10, 6))
sns.scatterplot(x=skew_df['SkewDiff'], y=raw_for_df[:len(skew_df['SkewDiff'])],
    edgecolor='k')
# plt.plot(skew_df['SkewDiff'], raw_for_df[:len(skew_df['Outlier'])], '.')

plt.xlabel('Skewness difference $I$')
plt.ylabel('FOR')

plt.savefig(os.path.join('plots', 'skew_vs_FOR.png'), dpi=600, bbox_inches='tight')
plt.savefig(os.path.join('plots', 'skew_vs_FOR.pdf'), dpi=600, bbox_inches='tight')
plt.show()
plt.close()

# %%
ks_df = pd.DataFrame()

for dataset in os.listdir(os.path.join('.', 'results_BINARYdata_fdr_0.10')):
    dataset_dir = os.path.join('.', 'results_BINARYdata_fdr_0.10', dataset)
    if not os.path.isdir(dataset_dir) or 'global' in dataset_dir:
        continue

    if os.path.exists(os.path.join(dataset_dir, 'pval_metrics_v2.csv')):
        df = pd.read_csv(os.path.join(dataset_dir, 'pval_metrics_v2.csv'))
        df = df[df.Metric == 'KSStat']
        df = df.assign(Dataset=dataset)

        ks_df = pd.concat([
            ks_df,
            df.pivot_table(values='Value', index='Dataset', columns=['Type'])
        ])

ks_df['KSStatDiff'] = ks_df.Outlier - ks_df.Inlier
display(ks_df)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
plt.figure(figsize=(10, 6))

norm = plt.Normalize(ks_df['KSStatDiff'].min(), ks_df['KSStatDiff'].max())
sm = plt.cm.ScalarMappable(norm=norm)
sns.scatterplot(x=skew_df['SkewDiff'], y=raw_for_df[:len(ks_df)],
    edgecolor='k', hue=ks_df['KSStatDiff'])
plt.colorbar(sm)
# plt.plot(skew_df['SkewDiff'], raw_for_df[:len(skew_df['Outlier'])], '.')

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
plt.figure(figsize=(10, 6))

norm = plt.Normalize(ks_df['KSStatDiff'].min(), ks_df['KSStatDiff'].max())
sm = plt.cm.ScalarMappable(norm=norm)
sns.scatterplot(x=skew_df['SkewDiff'], y=raw_for_df[:len(skew_df['SkewDiff'])],
    edgecolor='k', hue=ks_df['KSStatDiff'])
plt.colorbar(sm)
# plt.plot(skew_df['SkewDiff'], raw_for_df[:len(skew_df['Outlier'])], '.')


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
