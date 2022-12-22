# %%
import os
import numpy as np
import pandas as pd
from occ_all_tests_lib import *

test_type = 'fdr'
DATASET_TYPE = 'BINARYdata'
# DATASET_TYPE = 'SIMPLEtest'
alpha = 0.1
# alpha = 0.25

RESULTS_ROOT = os.path.join(
    'results'
)

for results_dir in os.listdir(RESULTS_ROOT):
    if not results_dir.startswith('results_'):
        continue

    # f'results_{DATASET_TYPE}_{test_type}_{alpha:.2f}'
    results_info = results_dir.split('_')
    DATASET_TYPE = results_info[1]
    test_type = results_info[2]
    alpha = results_info[3]
    DATASET_TYPE = results_info[1]

    raw_results_file = None
    for file in os.listdir(os.path.join(RESULTS_ROOT, results_dir)):
        if 'raw-results' in file:
            raw_results_file = file
            break

    df = pd.read_csv(os.path.join(RESULTS_ROOT, results_dir, raw_results_file))
    
    metric_list = df.columns[4:]
    alpha_metrics = [m for m in metric_list if 'alpha' in m]
    metric_list = [m for m in metric_list if 'alpha' not in m]

    df = fill_nan_values(df)
    DIR = os.path.join(RESULTS_ROOT, results_dir)
    aggregate_results(df, metric_list, alpha_metrics, DIR, DATASET_TYPE, alpha)

# %%
