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
import occ_datasets
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
resampling_repeats = 10
os.makedirs('results', exist_ok=True)

# datasets = [(dataset, 'mat') for dataset in occ_datasets.MAT_DATASETS] + \
#     [(dataset, 'arff') for dataset in occ_datasets.ARFF_DATASETS]

datasets = [
    # .mat
    ('Arrhythmia', 'mat'),
    ('Breastw', 'mat'),
    ('Cardio', 'mat'),
    ('Ionosphere', 'mat'),
    ('Lympho', 'mat'),
    ('Mammography', 'mat'),
    ('Optdigits', 'mat'),
    ('Pima', 'mat'),
    ('Satellite', 'mat'),
    ('Satimage-2', 'mat'),
    ('Shuttle', 'mat'),
    ('Speech', 'mat'),
    ('WBC', 'mat'),
    ('Wine', 'mat'),
    # .arff
    ('Arrhythmia', 'arff'),
    ('Cardiotocography', 'arff'),
    ('HeartDisease', 'arff'),
    ('Hepatitis', 'arff'),
    ('InternetAds', 'arff'),
    ('Ionosphere', 'arff'),
    ('KDDCup99', 'arff'),
    ('Lymphography', 'arff'),
    ('Pima', 'arff'),
    ('Shuttle', 'arff'),
    ('SpamBase', 'arff'),
    ('Stamps', 'arff'),
    ('Waveform', 'arff'),
    ('WBC', 'arff'),
    ('WDBC', 'arff'),
    ('WPBC', 'arff'),
]
# datasets = [
#     ('Speech', 'mat'),
#     # ('KDDCup99', 'arff'),
# ]

full_results = []

for (dataset, format) in datasets:
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
            for exp in range(n_repeats):
                # Load data
                X, y = occ_datasets.load_dataset(dataset, format)
                X_train, X_test, y_test = occ_datasets.split_occ_dataset(X, y, train_ratio=0.6)
                inlier_rate = np.mean(y_test)

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
                
                for cutoff_type in [
                    'Standard', # Empirical
                    'Chi-squared',
                    'Bootstrap',
                    'Multisplit'
                ]:
                    if cutoff_type != 'Standard' and not 'ECOD' in baseline:
                        continue
                    
                    N = len(X_train)
                    if cutoff_type == 'Bootstrap':
                        thresholds = []

                        for i in range(resampling_repeats):
                            bootstrap_samples = np.random.choice(range(N), size=N, replace=True)
                            is_bootstrap_sample = np.isin(range(N), bootstrap_samples)
                            X_boot_train, X_boot_cal = X_train[is_bootstrap_sample], X_train[~is_bootstrap_sample]
                            
                            clf.fit(X_boot_train)
                            scores = clf.score_samples(X_boot_cal)

                            emp_quantile = np.quantile(scores, q=1 - inlier_rate)
                            thresholds.append(emp_quantile)                        
                        threshold = np.mean(thresholds)
                    elif cutoff_type == 'Multisplit':
                        cal_scores_all = np.zeros((resampling_repeats, N - int(N/2)))
                        for i in range(resampling_repeats):
                            multisplit_samples = np.random.choice(range(N), size=int(N/2), replace=False)
                            is_multisplit_sample = np.isin(range(N), multisplit_samples)
                            X_multi_train, X_multi_cal = X_train[is_multisplit_sample], X_train[~is_multisplit_sample]
                            
                            clf.fit(X_multi_train)
                            cal_scores = clf.score_samples(X_multi_cal)
                            cal_scores_all[i, :] = cal_scores

                    try:
                        clf.fit(X_train)
                    except:
                        # LinAlgError("Singular matrix")
                        continue

                    scores = clf.score_samples(X_test)
                    auc = metrics.roc_auc_score(y_test, scores)

                    if cutoff_type == 'Standard':
                        emp_quantile = np.quantile(scores, q=1 - inlier_rate)
                        y_pred = np.where(scores > emp_quantile, 1, 0)
                    elif cutoff_type == 'Chi-squared':
                        d = X_test.shape[1]
                        chi_quantile = -scipy.stats.chi2.ppf(1 - inlier_rate, 2 * d)
                        y_pred = np.where(scores > chi_quantile, 1, 0)
                    elif cutoff_type == 'Bootstrap':
                        y_pred = np.where(scores > threshold, 1, 0)
                    elif cutoff_type == 'Multisplit':
                        p_vals_all = np.zeros((resampling_repeats, len(scores)))
                        for i in range(resampling_repeats):
                            cal_scores = cal_scores_all[i, :]
                            num_smaller_cal_scores = (scores > cal_scores.reshape(-1, 1)).sum(axis=0)
                            p_vals = (num_smaller_cal_scores + 1) / (len(cal_scores) + 1)
                            p_vals_all[i, :] = p_vals
                        p_vals = 2 * np.median(p_vals_all, axis=0)
                        y_pred = np.where(p_vals < 0.05, 0, 1)
                        # what should be the threshold?
                
                    acc = metrics.accuracy_score(y_test, y_pred)
                    pre = metrics.precision_score(y_test, y_pred)
                    rec = metrics.recall_score(y_test, y_pred)
                    f1 = metrics.f1_score(y_test, y_pred)

                    print(f'{dataset}: {baseline}{"+PCA" if use_PCA else ""} ({cutoff_type}, {exp+1}/{n_repeats})' + \
                        f' ||| AUC: {100 * auc:3.2f}, ACC: {100 * acc:3.2f}, F1: {100 * f1:3.2f}')
                    occ_metrics = {
                        'Dataset': dataset,
                        'Method': baseline + ("+PCA" if use_PCA else ""),
                        'Cutoff': cutoff_type,
                        'Exp': exp + 1,
                        'AUC': auc,
                        'Accuracy': acc,
                        'Precision': pre,
                        'Recall': rec,
                        'F1': f1,
                    }
                    results.append(occ_metrics)
                    full_results.append(occ_metrics)
    
    df = pd.DataFrame.from_records(results)

    dataset_df = df[df.Dataset == dataset]
    res_df = dataset_df.groupby(['Dataset', 'Method', 'Cutoff'])\
        [['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']] \
        .mean() \
        .round(4) \
        * 100
    display(res_df)
    res_df.to_csv(os.path.join('results', f'dataset-v3-{format}-{dataset}.csv'))

# Full result pivots
df = pd.DataFrame.from_records(full_results)
df
(df.loc[df.Cutoff == 'Standard'] \
    .pivot_table(values='AUC', index=['Dataset'], columns=['Method', 'Cutoff']) \
    .round(4) \
    * 100) \
    .to_csv(os.path.join('results', f'dataset-v3-all-AUC.csv'))
(df \
    .pivot_table(values='Accuracy', index=['Dataset'], columns=['Method', 'Cutoff']) \
    .round(4) \
    * 100) \
    .to_csv(os.path.join('results', f'dataset-v3-all-ACC.csv'))
(df \
    .pivot_table(values='F1', index=['Dataset'], columns=['Method', 'Cutoff']) \
    .round(4) \
    * 100) \
    .to_csv(os.path.join('results', f'dataset-v3-all-F1.csv'))
(df \
    .pivot_table(values='Precision', index=['Dataset'], columns=['Method', 'Cutoff']) \
    .round(4) \
    * 100) \
    .to_csv(os.path.join('results', f'dataset-v3-all-PRE.csv'))
(df \
    .pivot_table(values='Recall', index=['Dataset'], columns=['Method', 'Cutoff']) \
    .round(4) \
    * 100) \
    .to_csv(os.path.join('results', f'dataset-v3-all-REC.csv'))

# %%
