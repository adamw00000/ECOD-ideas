# %%
import os
import numpy as np
import pandas as pd
from occ_all_tests_lib import *

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
aggregate_results(df, metric_list, alpha_metrics, DIR, DATASET_TYPE, alpha)

# %%
