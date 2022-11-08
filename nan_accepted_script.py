# %%
import os
import pandas as pd
import numpy as np
from occ_all_tests_common import *


for dir in os.listdir('.'):
    if not dir.startswith('results_') and not dir.startswith('resultsdist_'):
        continue

    if dir.startswith('results_'):
        index_cols = 1
    else:
        index_cols = 3

    alpha = float(dir.split('_')[2])

    for file in os.listdir(dir):
        if not file.endswith('all-FDR.csv'):
            continue

        print(os.path.join(dir, file))
    
        df = pd.read_csv(os.path.join(dir, file), header=[0, 1], index_col=[0, 1, 2][:index_cols])
        fdr_df = df[:len(df) - 1].astype(float)

        fdr_alpha_df = (fdr_df < alpha) | (np.isnan(fdr_df))
        fdr_alpha_df = append_mean_row(fdr_alpha_df)
        display(fdr_alpha_df)


        os.makedirs(os.path.join('nan_accepted', dir), exist_ok=True)
        fdr_alpha_df.to_csv(os.path.join('nan_accepted', dir, file.replace('FDR', 'FDR-alpha')))

# %%
