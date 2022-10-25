# %%
from sklearn.decomposition import PCA

def PCA_by_variance(X_train, X_test, variance_threshold=0.9):
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train)

    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    are_features_enough = explained_variance >= variance_threshold
    num_features = np.where(are_features_enough)[0][0] + 1 if np.any(are_features_enough) else X.shape[1]
    X_train_pca = X_train_pca[:, :num_features]

    X_test_pca = pca.transform(X_test)
    X_test_pca = X_test_pca[:, :num_features]
    return X_train_pca, X_test_pca, explained_variance[:num_features]

import numpy as np
from scipy.spatial.distance import cdist, euclidean

def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1

class GeomMedianDistance():
    def fit(self, X, eps=1e-5):
        self.median = geometric_median(X, eps)
        return self
    
    def score_samples(self, X):
        return -np.linalg.norm(X - self.median, 2, axis=1)

import numpy as np

def dot_diag(A, B):
    # Diagonal of the matrix product
    # equivalent to: np.diag(A @ B)
    return np.einsum('ij,ji->i', A, B)

class Mahalanobis():
    def fit(self, X):
        self.mu = np.mean(X, axis=0).reshape(1, -1)
        self.sigma_inv = np.linalg.inv(np.cov(X.T))
        return self
    
    def score_samples(self, X):
        # (X - self.mu) @ self.sigma_inv @ (X - self.mu).T
        # but we need only the diagonal
        mahal = dot_diag((X - self.mu) @ self.sigma_inv, (X - self.mu).T)
        return 1 / (1 + mahal)

from pyod.models.ecod import ECOD
from ecod_v2 import ECODv2

class PyODWrapper():
    def __init__(self, model):
        self.model = model
    
    def fit(self, X_train):
        self.model.fit(X_train)
        return self

    def score_samples(self, X):
        return -self.model.decision_function(X)

# %%
import os
import datasets
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

from pyod.models.ecod import ECOD
from ecod_v2 import ECODv2
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

n_repeats = 10
os.makedirs('results', exist_ok=True)

def invert_labels_for_occ(y):
    if np.mean(y) < 0.5:
        y = np.where(y == 1, 0, 1)
    return y

def split_dataset(X, y, train_ratio=0.6):
    idx_pos = np.where(y == 1)[0]
    idx_train = np.random.choice(idx_pos, int(train_ratio * len(idx_pos)))

    idx_test = np.ones_like(y, dtype=np.bool) # array of True
    idx_test[idx_train] = False

    X_train = X[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]

    return X_train, X_test, y_test


for dataset in datasets.DATASET_NAMES:
    results = []

    for baseline in [
        'ECOD',
        'ECODv2',
        'GeomMedian',
        'Mahalanobis',
        'OC-SVM',
        'IForest',
    ]:
        for use_PCA in [False, True]:
            for i in range(n_repeats):
                # Load data
                X, y = datasets.load_dataset(dataset)
                X_train, X_test, y_test = split_dataset(X, y, train_ratio=0.6)
                y = invert_labels_for_occ(y)

                if use_PCA:
                    X_train, X_test, _ = PCA_by_variance(X_train, X_test, variance_threshold=0.9)

                if use_PCA and not 'ECOD' in baseline:
                    continue
                if baseline == 'ECOD':
                    clf = PyODWrapper(ECOD())
                elif baseline == 'ECODv2':
                    clf = PyODWrapper(ECODv2())
                elif baseline == 'GeomMedian':
                    clf = GeomMedianDistance()
                elif baseline == 'Mahalanobis':
                    clf = Mahalanobis()
                elif baseline == 'OC-SVM':
                    clf = OneClassSVM()
                elif baseline == 'IForest':
                    clf = IsolationForest()
                
                try:
                    clf.fit(X_train)
                except:
                    # LinAlgError("Singular matrix")
                    continue

                scores = clf.score_samples(X_test)
                auc = metrics.roc_auc_score(y_test, scores)

                inlier_rate = np.mean(y_test)

                for cutoff_type in [
                    'Empirical',
                    'Chi-squared',
                    # 'Bootstrap',
                    # 'Multisplit',
                ]:
                    if cutoff_type != 'Empirical' and not 'ECOD' in baseline:
                        continue

                    if cutoff_type == 'Empirical':
                        emp_quantile = np.quantile(scores, q=1 - inlier_rate)
                        y_pred = np.where(scores > emp_quantile, 1, 0)
                    elif cutoff_type == 'Chi-squared':
                        d = X_test.shape[1]
                        chi_quantile = -scipy.stats.chi2.ppf(1 - inlier_rate, 2 * d)
                        y_pred = np.where(scores > chi_quantile, 1, 0)
                    elif cutoff_type == 'Bootstrap':
                        pass
                    elif cutoff_type == 'Multisplit':
                        pass
                
                    acc = metrics.accuracy_score(y_test, y_pred)

                    print(f'{dataset}: {baseline}{"+PCA" if use_PCA else ""} ({cutoff_type}, {i+1}/{n_repeats})' + \
                        f' ||| AUC: {100 * auc:3.2f}, ACC: {100 * acc:3.2f}')
                    results.append({
                        'Dataset': dataset,
                        'Method': baseline + ("+PCA" if use_PCA else ""),
                        'Cutoff': cutoff_type,
                        'Exp': i + 1,
                        'AUC': auc,
                        'Accuracy': acc,
                    })
    
    df = pd.DataFrame.from_records(results)

    dataset_df = df[df.Dataset == dataset]
    res_df = dataset_df.groupby(['Dataset', 'Method', 'Cutoff'])\
        [['AUC', 'Accuracy']] \
        .mean() \
        .round(4) \
        * 100
    display(res_df)
    res_df.to_csv(os.path.join('results', f'dataset-{dataset}.csv'))

# %%
