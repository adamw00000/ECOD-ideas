# %%
from occ_tests_common import *

import os
import occ_datasets
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

from pyod.models.ecod import ECOD
from ecod_v2 import ECODv2
from ecod_v2_min import ECODv2Min
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

    for exp in range(n_repeats):
        # Load data
        X, y = occ_datasets.load_dataset(dataset, format)
        X_train_orig, X_test_orig, y_test_orig = occ_datasets.split_occ_dataset(X, y, train_ratio=0.6)
        inlier_rate = np.mean(y_test_orig)

        for baseline in [
            'ECOD',
            'ECODv2',
            'ECODv2Min',
            'GeomMedian',
            'Mahalanobis',
            'OC-SVM',
            'IForest',
        ]:
            for pca_variance_threshold in [0.5, 0.9, 1.0, None]:
                X_train, X_test, y_test = X_train_orig, X_test_orig, y_test_orig
                if pca_variance_threshold is not None:
                    if not 'ECODv2' in baseline:
                        continue
                    X_train, X_test, _ = PCA_by_variance(X_train, X_test, pca_variance_threshold)

                if baseline == 'ECOD':
                    clf = PyODWrapper(ECOD())
                elif baseline == 'ECODv2':
                    clf = PyODWrapper(ECODv2())
                elif baseline == 'ECODv2Min':
                    clf = PyODWrapper(ECODv2Min())
                elif baseline == 'GeomMedian':
                    clf = GeomMedianDistance()
                elif baseline == 'Mahalanobis':
                    clf = Mahalanobis()
                elif baseline == 'OC-SVM':
                    clf = OneClassSVM()
                elif baseline == 'IForest':
                    clf = IsolationForest()
                
                for cutoff_type in [
                    'Empirical',
                    'Chi-squared',
                    'Bootstrap',
                    'Multisplit'
                ]:
                    if cutoff_type != 'Empirical' and not 'ECODv2' in baseline and not 'Mahalanobis' in baseline:
                        continue
                    
                    N = len(X_train)
                    if cutoff_type == 'Bootstrap' or cutoff_type == 'Multisplit':
                        thresholds = []

                        for i in range(resampling_repeats):
                            if cutoff_type == 'Bootstrap':
                                resampling_samples = np.random.choice(range(N), size=N, replace=True)
                            else:
                                # Multisplit
                                resampling_samples = np.random.choice(range(N), size=int(N/2), replace=False)
                            is_selected_sample = np.isin(range(N), resampling_samples)
                            X_resampling_train, X_resampling_cal = X_train[is_selected_sample], X_train[~is_selected_sample]
                            
                            clf.fit(X_resampling_train)
                            scores = clf.score_samples(X_resampling_cal)

                            emp_quantile = np.quantile(scores, q=1 - inlier_rate)
                            thresholds.append(emp_quantile)                        
                        resampling_threshold = np.mean(thresholds)

                    clf.fit(X_train)

                    scores = clf.score_samples(X_test)
                    auc = metrics.roc_auc_score(y_test, scores)

                    if cutoff_type == 'Empirical':
                        emp_quantile = np.quantile(scores, q=1 - inlier_rate)
                        y_pred = np.where(scores > emp_quantile, 1, 0)
                    elif cutoff_type == 'Chi-squared':
                        d = X_test.shape[1]
                        chi_quantile = -scipy.stats.chi2.ppf(1 - inlier_rate, 2 * d)
                        y_pred = np.where(scores > chi_quantile, 1, 0)
                    elif cutoff_type == 'Bootstrap' or cutoff_type == 'Multisplit':
                        y_pred = np.where(scores > resampling_threshold, 1, 0)
                
                    false_detections = np.sum((y_pred == 0) & (y_test == 1))
                    detections = np.sum(y_pred == 0)
                    fdr = false_detections / detections

                    acc = metrics.accuracy_score(y_test, y_pred)
                    pre = metrics.precision_score(y_test, y_pred)
                    rec = metrics.recall_score(y_test, y_pred)
                    f1 = metrics.f1_score(y_test, y_pred)

                    print(f'{dataset}.{format}: {baseline}{f"+PCA{pca_variance_threshold:.1f}" if pca_variance_threshold is not None else ""} ({cutoff_type}, {exp+1}/{n_repeats})' + \
                        f' ||| AUC: {100 * auc:3.2f}, ACC: {100 * acc:3.2f}, F1: {100 * f1:3.2f}')
                    occ_metrics = {
                        'Dataset': f'({format}) {dataset}',
                        'Method': baseline + (f"+PCA{pca_variance_threshold:.1f}" if pca_variance_threshold is not None else ""),
                        'Cutoff': cutoff_type,
                        'Exp': exp + 1,
                        'AUC': auc,
                        'ACC': acc,
                        'PRE': pre,
                        'REC': rec,
                        'F1': f1,
                        'FDR': fdr,
                    }
                    results.append(occ_metrics)
                    full_results.append(occ_metrics)
    
    df = pd.DataFrame.from_records(results)

    dataset_df = df[df.Dataset == f'({format}) {dataset}']
    res_df = dataset_df.groupby(['Dataset', 'Method', 'Cutoff'])\
        [['AUC', 'ACC', 'PRE', 'REC', 'F1', 'FDR']] \
        .mean() \
        .round(4) \
        * 100

    res_df = append_mean_row(res_df)
    display(res_df)
    res_df.to_csv(os.path.join('results', f'dataset-{format}-{dataset}.csv'))

# Full result pivots
df = pd.DataFrame.from_records(full_results)
df

for metric in ['AUC', 'ACC', 'F1', 'PRE', 'REC', 'FDR']:
    metric_df = df
    if metric == 'AUC':
        metric_df = df.loc[df.Cutoff == 'Empirical']
    
    pivot = metric_df \
        .pivot_table(values=metric, index=['Dataset'], columns=['Method', 'Cutoff'], dropna=False) \
        * 100
    
    pivot = append_mean_row(pivot)
    pivot \
        .round(2) \
        .to_csv(os.path.join('results', f'dataset-all-{metric}.csv'))

# %%
