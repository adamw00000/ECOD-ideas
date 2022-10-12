# %%
import os
import scipy.io
import numpy as np
import pandas as pd

# https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/
ARFF_DATASETS = {
    # literature
    'ALOI': 'ALOI.arff',
    'Glass': 'Glass.arff',
    'Ionosphere': 'Ionosphere.arff',
    'KDDCup99': 'KDDCup99.arff',
    'Lymphography': 'Lymphography.arff',
    'PenDigits': 'PenDigits.arff',
    'Shuttle': 'Shuttle.arff',
    'Waveform': 'Waveform.arff',
    'WBC': 'WBC.arff',
    'WDBC': 'WDBC.arff',
    'WPBC': 'WPBC.arff',
    # semantic
    'Annthyroid': 'Annthyroid.arff',
    'Arrhythmia': 'Arrhythmia.arff',
    'Cardiotocography': 'Cardiotocography.arff',
    'HeartDisease': 'HeartDisease.arff',
    'Hepatitis': 'Hepatitis.arff',
    'InternetAds': 'InternetAds.arff',
    'PageBlocks': 'PageBlocks.arff',
    'Parkinson': 'Parkinson.arff',
    'Pima': 'Pima.arff',
    'Stamps': 'Stamps.arff',
    'SpamBase': 'SpamBase.arff',
    'Wilt': 'Wilt.arff',
}

# http://odds.cs.stonybrook.edu/
MAT_DATASETS = {
    'Lympho': 'lympho.mat',
    'WBC': 'wbc.mat',
    'Glass': 'glass.mat',
    'Vowels': 'vowels.mat',
    'Cardio': 'cardio.mat',
    'Thyroid': 'thyroid.mat',
    'Musk': 'musk.mat',
    'Satimage-2': 'satimage-2.mat',
    'Letter Recognition': 'letter.mat',
    'Speech': 'speech.mat',
    'Pima': 'pima.mat',
    'Satellite': 'satellite.mat',
    'Shuttle': 'shuttle.mat',
    'Breastw': 'breastw.mat',
    'Arrhythmia': 'arrhythmia.mat',
    'Ionosphere': 'ionosphere.mat',
    'Mnist': 'mnist.mat',
    'Optdigits': 'optdigits.mat',
    'Http (KDDCUP99)': 'http.mat',
    'ForestCover': 'cover.mat',
    'Smtp (KDDCUP99)': 'smtp.mat',
    'Mammography': 'mammography.mat',
    'Annthyroid': 'annthyroid.mat',
    'Pendigits': 'pendigits.mat',
    'Ecoli': 'ecoli.mat',
    'Wine': 'wine.mat',
    'Vertebral': 'vertebral.mat',
}

def load_dataset(name, format):
    if format == 'arff':
        if name not in ARFF_DATASETS:
            raise ValueError('No such dataset')
        
        filename = ARFF_DATASETS[name]
        data = scipy.io.arff.loadarff(os.path.join('occ-data', filename))
        df = pd.DataFrame(data[0])

        X, y = df.drop(columns=['id', 'outlier']).to_numpy(), df['outlier'].to_numpy()
        y = np.where(y == b'yes', 0, 1) # y means "is inlier"
    elif format == 'mat':
        if name not in MAT_DATASETS:
            raise ValueError('No such dataset')
        
        filename = MAT_DATASETS[name]
        data = scipy.io.loadmat(os.path.join('occ-data', filename))

        X, y = data['X'], data['y'].reshape(-1)
        y = np.where(y == 1, 0, 1) # y means "is inlier"
    else:
        raise ValueError("Format not supported; choose 'arff' or 'mat'")

    if name == 'Shuttle':
        # https://arxiv.org/pdf/2201.00382.pdf
        X, y = X[:10000], y[:10000]
    
    X = X.astype(np.float)

    return X, y


def split_occ_dataset(X, y, train_ratio=0.6):
    idx_pos = np.where(y == 1)[0]
    idx_train = np.random.choice(idx_pos, int(train_ratio * len(idx_pos)))

    idx_test = np.ones_like(y, dtype=np.bool) # array of True
    idx_test[idx_train] = False

    X_train = X[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]

    return X_train, X_test, y_test

