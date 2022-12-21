# %%
import os
import pandas as pd
import numpy as np
import shutil

OLD_RES_DIR = 'results'
NEW_RES_DIR = 'results-filtered-all'
cutoffs = ['Empirical', 'Multisplit',
    'Multisplit+BH', 'Multisplit+BH+pi', 'Multisplit+FOR-CTL-old',
    'NoSplit', 'Multisplit_1-repeat']
methods = [
    'A^3',
    'ECODv2',
    'ECODv2+PCA1.0',
    'IForest',
    'Mahalanobis',
    'OracleLR',
    'OracleRF',
]

for dir in os.listdir(os.path.join(OLD_RES_DIR)):
    if '0.25' in dir:
        continue
    
    os.makedirs(os.path.join(NEW_RES_DIR, dir), exist_ok=True)
    for entry in os.listdir(os.path.join(OLD_RES_DIR, dir)):
        if entry == 'global-plots':
            os.makedirs(os.path.join(NEW_RES_DIR, dir, entry), exist_ok=True)
            for image_file in os.listdir(os.path.join(OLD_RES_DIR, dir, entry)):
                if ('BH' in image_file or 'FOR-CTL-old' in image_file) \
                        and ('A^3' in image_file or 'ECODv2' in image_file or 'IForest' in image_file or 'Mahalanobis' in image_file or 'OracleLR' in image_file or 'OracleRF' in image_file):
                    shutil.copy(
                        os.path.join(OLD_RES_DIR, dir, entry, image_file),
                        os.path.join(NEW_RES_DIR, dir, entry, image_file),
                    )
        elif os.path.isdir(os.path.join(OLD_RES_DIR, dir, entry)):
            os.makedirs(os.path.join(NEW_RES_DIR, dir, entry), exist_ok=True)
            for dataset_entry in os.listdir(os.path.join(OLD_RES_DIR, dir, entry)):
                if os.path.isdir(os.path.join(OLD_RES_DIR, dir, entry, dataset_entry)):
                    os.makedirs(os.path.join(NEW_RES_DIR, dir, entry, dataset_entry), exist_ok=True)
                    for image_file in os.listdir(os.path.join(OLD_RES_DIR, dir, entry, dataset_entry)):
                        if ('BH' in image_file or 'FOR-CTL-old' in image_file) \
                                and ('A^3' in image_file or 'ECODv2' in image_file or 'IForest' in image_file or 'Mahalanobis' in image_file or 'OracleLR' in image_file or 'OracleRF' in image_file):
                            shutil.copy(
                                os.path.join(OLD_RES_DIR, dir, entry, dataset_entry, image_file),
                                os.path.join(NEW_RES_DIR, dir, entry, dataset_entry, image_file),
                            )
                
                if dataset_entry.endswith('.csv'):
                    df = pd.read_csv(os.path.join(OLD_RES_DIR, dir, entry, dataset_entry))
                    if 'aggregate' in dataset_entry:
                        df['BFOR'] = df['#FN'] / (df['#TN'] + df['#FN']) # FN / PN
                        df['BFOR < alpha'] = df['BFOR'] < 0.1
                        df.to_csv(os.path.join(OLD_RES_DIR, dir, entry, dataset_entry), index=False)
                    df = df.loc[np.isin(df.Cutoff, cutoffs) & np.isin(df.Method, methods)]
                    df.to_csv(os.path.join(NEW_RES_DIR, dir, entry, dataset_entry), index=False)
        elif '-all-' in entry and 'transposed' not in entry:
            df = pd.read_csv(os.path.join(OLD_RES_DIR, dir, entry), header=[0, 1], index_col=0)
            df = df[[col for col in df.columns if col[0] in methods and col[1] in cutoffs]]
            df.to_csv(os.path.join(NEW_RES_DIR, dir, entry))

            if '#TN' in entry:
                tn_df = df
            elif '#FN' in entry:
                fn_df = df
        else:
            df = pd.read_csv(os.path.join(OLD_RES_DIR, dir, entry))
            if '-all' not in entry and '-raw' not in entry:
                df['BFOR'] = df['#FN'] / (df['#TN'] + df['#FN']) # FN / PN
                df['BFOR < alpha'] = df['BFOR'] < 0.1
                df.to_csv(os.path.join(OLD_RES_DIR, dir, entry), index=False)
            df = df.loc[np.isin(df.Cutoff, cutoffs) & np.isin(df.Method, methods)]
            df.to_csv(os.path.join(NEW_RES_DIR, dir, entry), index=False)

    bfor_df = fn_df / (tn_df + fn_df)
    bfor_alpha_df = (bfor_df < 0.1)

    bfor_df.iloc[-1] = bfor_df.iloc[:-1].mean()
    bfor_df = bfor_df.round(3)
    bfor_df.to_csv(os.path.join(NEW_RES_DIR, dir, 'BINARYdata-all-BFOR.csv'))
    bfor_df.transpose().to_csv(os.path.join(NEW_RES_DIR, dir, 'BINARYdata-all-BFOR-transposed.csv'))

    bfor_alpha_df.iloc[-1] = bfor_alpha_df.iloc[:-1].mean().round(3)
    bfor_alpha_df.to_csv(os.path.join(NEW_RES_DIR, dir, 'BINARYdata-all-BFOR-alpha.csv'))
    bfor_alpha_df.transpose().to_csv(os.path.join(NEW_RES_DIR, dir, 'BINARYdata-all-BFOR-alpha-transposed.csv'))

# %%
import os
import pandas as pd
import numpy as np
import shutil

OLD_RES_DIR = 'results'
NEW_RES_DIR = 'results-filtered-for-ctl'
cutoffs = ['Empirical', 'Multisplit+FOR-CTL-old']
methods = [
    'A^3',
    'ECODv2',
    'ECODv2+PCA1.0',
    'IForest',
    'Mahalanobis',
    'OracleLR',
    'OracleRF',
]

for dir in os.listdir(os.path.join(OLD_RES_DIR)):
    if '0.25' in dir:
        continue
    
    os.makedirs(os.path.join(NEW_RES_DIR, dir), exist_ok=True)
    for entry in os.listdir(os.path.join(OLD_RES_DIR, dir)):
        if entry == 'global-plots':
            os.makedirs(os.path.join(NEW_RES_DIR, dir, entry), exist_ok=True)
            for image_file in os.listdir(os.path.join(OLD_RES_DIR, dir, entry)):
                if ('FOR-CTL-old' in image_file) \
                        and ('A^3' in image_file or 'ECODv2' in image_file or 'IForest' in image_file or 'Mahalanobis' in image_file or 'OracleLR' in image_file or 'OracleRF' in image_file):
                    shutil.copy(
                        os.path.join(OLD_RES_DIR, dir, entry, image_file),
                        os.path.join(NEW_RES_DIR, dir, entry, image_file),
                    )
        elif os.path.isdir(os.path.join(OLD_RES_DIR, dir, entry)):
            os.makedirs(os.path.join(NEW_RES_DIR, dir, entry), exist_ok=True)
            for dataset_entry in os.listdir(os.path.join(OLD_RES_DIR, dir, entry)):
                if os.path.isdir(os.path.join(OLD_RES_DIR, dir, entry, dataset_entry)):
                    os.makedirs(os.path.join(NEW_RES_DIR, dir, entry, dataset_entry), exist_ok=True)
                    for image_file in os.listdir(os.path.join(OLD_RES_DIR, dir, entry, dataset_entry)):
                        if ('FOR-CTL-old' in image_file) \
                                and ('A^3' in image_file or 'ECODv2' in image_file or 'IForest' in image_file or 'Mahalanobis' in image_file or 'OracleLR' in image_file or 'OracleRF' in image_file):
                            shutil.copy(
                                os.path.join(OLD_RES_DIR, dir, entry, dataset_entry, image_file),
                                os.path.join(NEW_RES_DIR, dir, entry, dataset_entry, image_file),
                            )
                
                if dataset_entry.endswith('.csv'):
                    df = pd.read_csv(os.path.join(OLD_RES_DIR, dir, entry, dataset_entry))
                    if 'aggregate' in dataset_entry:
                        df['BFOR'] = df['#FN'] / (df['#TN'] + df['#FN']) # FN / PN
                        df['BFOR < alpha'] = df['BFOR'] < 0.1
                        df.to_csv(os.path.join(OLD_RES_DIR, dir, entry, dataset_entry), index=False)
                    df = df.loc[np.isin(df.Cutoff, cutoffs) & np.isin(df.Method, methods)]
                    df.to_csv(os.path.join(NEW_RES_DIR, dir, entry, dataset_entry), index=False)
        elif '-all-' in entry and 'transposed' not in entry:
            df = pd.read_csv(os.path.join(OLD_RES_DIR, dir, entry), header=[0, 1], index_col=0)
            df = df[[col for col in df.columns if col[0] in methods and col[1] in cutoffs]]
            df.to_csv(os.path.join(NEW_RES_DIR, dir, entry))

            if '#TN' in entry:
                tn_df = df
            elif '#FN' in entry:
                fn_df = df
        else:
            df = pd.read_csv(os.path.join(OLD_RES_DIR, dir, entry))
            if '-all' not in entry and '-raw' not in entry:
                df['BFOR'] = df['#FN'] / (df['#TN'] + df['#FN']) # FN / PN
                df['BFOR < alpha'] = df['BFOR'] < 0.1
                df.to_csv(os.path.join(OLD_RES_DIR, dir, entry), index=False)
            df = df.loc[np.isin(df.Cutoff, cutoffs) & np.isin(df.Method, methods)]
            df.to_csv(os.path.join(NEW_RES_DIR, dir, entry), index=False)

    bfor_df = fn_df / (tn_df + fn_df)
    bfor_alpha_df = (bfor_df < 0.1)

    bfor_df.iloc[-1] = bfor_df.iloc[:-1].mean()
    bfor_df = bfor_df.round(3)
    bfor_df.to_csv(os.path.join(NEW_RES_DIR, dir, 'BINARYdata-all-BFOR.csv'))
    bfor_df.transpose().to_csv(os.path.join(NEW_RES_DIR, dir, 'BINARYdata-all-BFOR-transposed.csv'))

    bfor_alpha_df.iloc[-1] = bfor_alpha_df.iloc[:-1].mean().round(3)
    bfor_alpha_df.to_csv(os.path.join(NEW_RES_DIR, dir, 'BINARYdata-all-BFOR-alpha.csv'))
    bfor_alpha_df.transpose().to_csv(os.path.join(NEW_RES_DIR, dir, 'BINARYdata-all-BFOR-alpha-transposed.csv'))

# %%
import os
import pandas as pd
import numpy as np
import shutil

OLD_RES_DIR = 'results'
NEW_RES_DIR = 'results-filtered-fdr-ctl'
cutoffs = ['Empirical', 'Multisplit+BH', 'Multisplit+BH+pi']
methods = [
    'A^3',
    'ECODv2',
    'ECODv2+PCA1.0',
    'IForest',
    'Mahalanobis',
    'OracleLR',
    'OracleRF',
]

for dir in os.listdir(os.path.join(OLD_RES_DIR)):
    if '0.25' in dir:
        continue

    os.makedirs(os.path.join(NEW_RES_DIR, dir), exist_ok=True)
    for entry in os.listdir(os.path.join(OLD_RES_DIR, dir)):
        if entry == 'global-plots':
            os.makedirs(os.path.join(NEW_RES_DIR, dir, entry), exist_ok=True)
            for image_file in os.listdir(os.path.join(OLD_RES_DIR, dir, entry)):
                if ('BH' in image_file) \
                        and ('A^3' in image_file or 'ECODv2' in image_file or 'IForest' in image_file or 'Mahalanobis' in image_file or 'OracleLR' in image_file or 'OracleRF' in image_file):
                    shutil.copy(
                        os.path.join(OLD_RES_DIR, dir, entry, image_file),
                        os.path.join(NEW_RES_DIR, dir, entry, image_file),
                    )
        elif os.path.isdir(os.path.join(OLD_RES_DIR, dir, entry)):
            os.makedirs(os.path.join(NEW_RES_DIR, dir, entry), exist_ok=True)
            for dataset_entry in os.listdir(os.path.join(OLD_RES_DIR, dir, entry)):
                if os.path.isdir(os.path.join(OLD_RES_DIR, dir, entry, dataset_entry)):
                    os.makedirs(os.path.join(NEW_RES_DIR, dir, entry, dataset_entry), exist_ok=True)
                    for image_file in os.listdir(os.path.join(OLD_RES_DIR, dir, entry, dataset_entry)):
                        if ('BH' in image_file) \
                                and ('A^3' in image_file or 'ECODv2' in image_file or 'IForest' in image_file or 'Mahalanobis' in image_file or 'OracleLR' in image_file or 'OracleRF' in image_file):
                            shutil.copy(
                                os.path.join(OLD_RES_DIR, dir, entry, dataset_entry, image_file),
                                os.path.join(NEW_RES_DIR, dir, entry, dataset_entry, image_file),
                            )
                
                if dataset_entry.endswith('.csv'):
                    df = pd.read_csv(os.path.join(OLD_RES_DIR, dir, entry, dataset_entry))
                    if 'aggregate' in dataset_entry:
                        df['BFOR'] = df['#FN'] / (df['#TN'] + df['#FN']) # FN / PN
                        df['BFOR < alpha'] = df['BFOR'] < 0.1
                        df.to_csv(os.path.join(OLD_RES_DIR, dir, entry, dataset_entry), index=False)
                    df = df.loc[np.isin(df.Cutoff, cutoffs) & np.isin(df.Method, methods)]
                    df.to_csv(os.path.join(NEW_RES_DIR, dir, entry, dataset_entry), index=False)
        elif '-all-' in entry and 'transposed' not in entry:
            df = pd.read_csv(os.path.join(OLD_RES_DIR, dir, entry), header=[0, 1], index_col=0)
            df = df[[col for col in df.columns if col[0] in methods and col[1] in cutoffs]]
            df.to_csv(os.path.join(NEW_RES_DIR, dir, entry))

            if '#TN' in entry:
                tn_df = df
            elif '#FN' in entry:
                fn_df = df
        else:
            df = pd.read_csv(os.path.join(OLD_RES_DIR, dir, entry))
            if '-all' not in entry and '-raw' not in entry:
                df['BFOR'] = df['#FN'] / (df['#TN'] + df['#FN']) # FN / PN
                df['BFOR < alpha'] = df['BFOR'] < 0.1
                df.to_csv(os.path.join(OLD_RES_DIR, dir, entry), index=False)
            df = df.loc[np.isin(df.Cutoff, cutoffs) & np.isin(df.Method, methods)]
            df.to_csv(os.path.join(NEW_RES_DIR, dir, entry), index=False)

    bfor_df = fn_df / (tn_df + fn_df)
    bfor_alpha_df = (bfor_df < 0.1)

    bfor_df.iloc[-1] = bfor_df.iloc[:-1].mean()
    bfor_df = bfor_df.round(3)
    bfor_df.to_csv(os.path.join(NEW_RES_DIR, dir, 'BINARYdata-all-BFOR.csv'))
    bfor_df.transpose().to_csv(os.path.join(NEW_RES_DIR, dir, 'BINARYdata-all-BFOR-transposed.csv'))

    bfor_alpha_df.iloc[-1] = bfor_alpha_df.iloc[:-1].mean().round(3)
    bfor_alpha_df.to_csv(os.path.join(NEW_RES_DIR, dir, 'BINARYdata-all-BFOR-alpha.csv'))
    bfor_alpha_df.transpose().to_csv(os.path.join(NEW_RES_DIR, dir, 'BINARYdata-all-BFOR-alpha-transposed.csv'))


# %%
df = pd.read_csv(os.path.join(OLD_RES_DIR, 'results_BINARYdata_fdr_0.10', 'BINARYdata-all-FOR.csv'), header=[0, 1])
df

# %%
