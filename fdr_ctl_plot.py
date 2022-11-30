# %%
import os
import numpy as np
import pandas as pd
from occ_all_tests_common import *

test_type = 'fdr'
DATASET_TYPE = 'BINARYdata'
alpha = 0.05

DIR = os.path.join('results-partial', f'results_{DATASET_TYPE}_{test_type}_{alpha}')

df = pd.DataFrame()
for file in os.listdir(os.path.join(DIR)):
    if 'all' in file \
            or not file.endswith('.csv'):
        continue

    df = pd.concat([
        df,
        pd.read_csv(os.path.join(DIR, file))
    ])

df = df.reset_index(drop=True)
df = df[df.Dataset != 'Mean']
df[metrics_to_multiply_by_100] = df[metrics_to_multiply_by_100] / 100

metric_list = df.columns[3:]
alpha_metrics = [m for m in metric_list if 'alpha' in m]
metric_list = [m for m in metric_list if 'alpha' not in m]

df

# %%
metric = 'FDR'
controlling_cutoff = 'Multisplit+BH+pi'

metric_df = df[['Dataset', 'Method', 'Cutoff', metric]]
metric_df = metric_df[metric_df.Cutoff == controlling_cutoff]

metric_df

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

for method in metric_df.Method.unique():
    method_df = metric_df[metric_df.Method == method]

    datasets = df.Dataset.unique()
    plt.figure(figsize=(10, 8))
    sns.lineplot(data=method_df, x='Dataset', y=metric, hue='Cutoff')
    sns.lineplot(x=datasets, y=[alpha] * len(datasets), 
        hue=[f'alpha = {alpha:.2f}'] * len(datasets),
        style=[f'alpha = {alpha:.2f}'] * len(datasets),
        palette = ['k'], dashes=True)
    plt.title(f'{metric} ({method})')
    plt.xticks(rotation = 90)
    plt.ylim(-0.005, metric_df[metric].max())
    plt.show()
    plt.close()

# %%
