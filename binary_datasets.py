# %%
import os
import numpy as np
import pandas as pd

# https://datahub.io/machine-learning/
CSV_DATASETS = {
    x: f'{x}.csv' for x in [
        'Abalone',
        'Arrhythmia',
        'Banknote-authentication',
        'Breast-w',
        'Dermatology',
        'Diabetes',
        'Fertility',
        'Gas-drift',
        'Glass',
        'Haberman',
        'Heart-statlog',
        'Ionosphere',
        'Isolet',
        'Jm1',
        'Kc1',
        'Madelon',
        'Musk',
        'Optdigits',
        'Pendigits',
        'Satimage',
        'Segment',
        'Seismic-bumps',
        'Semeion',
        'Sonar',
        'Spambase',
        'Tic-tac-toe',
        'Vehicle',
        'Waveform-5000',
        'Wdbc',
        'Yeast',
    ]
}

full_dataset_list = [
    (x, 'csv') for x in [
        'Abalone',
        'Arrhythmia',
        'Banknote-authentication',
        'Breast-w',
        'Dermatology',
        'Diabetes',
        'Fertility',
        'Gas-drift',
        'Glass',
        'Haberman',
        'Heart-statlog',
        'Ionosphere',
        'Isolet',
        'Jm1',
        'Kc1',
        'Madelon',
        'Musk',
        'Optdigits',
        'Pendigits',
        'Satimage',
        'Segment',
        'Seismic-bumps',
        'Semeion',
        'Sonar',
        'Spambase',
        'Tic-tac-toe',
        'Vehicle',
        'Waveform-5000',
        'Wdbc',
        'Yeast',
    ]
]

def load_dataset(name, format):
    if format == 'csv':
        if name not in CSV_DATASETS:
            raise ValueError('No such dataset')
        
        filename = CSV_DATASETS[name]
        df = pd.read_csv(os.path.join('binary-data', filename))

        X, y = df.drop(columns=['Class', 'BinClass']).to_numpy(), df['BinClass'].to_numpy() # y means "is inlier"
    else:
        raise ValueError("Format not supported; choose 'csv'")
    
    X = X.astype(np.float)
    return X, y


def split_occ_dataset(X, y, train_ratio=0.6):
    idx_pos = np.where(y == 1)[0]
    idx_train = np.random.choice(idx_pos, int(train_ratio * len(idx_pos)), replace=False)

    idx_test = np.ones_like(y, dtype=np.bool) # array of True
    idx_test[idx_train] = False

    X_train = X[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]

    return X_train, X_test, y_test
