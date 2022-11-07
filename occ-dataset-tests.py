# %%
from occ_tests_common import *

import os
import occ_datasets
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

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
metric_list = ['AUC', 'ACC', 'PRE', 'REC', 'F1', 'FDR', 'FRR', 'T1E']

for (dataset, format) in datasets:
    print(f'({format}) {dataset}')
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
            for pca_variance_threshold in [None, 1.0]:
                if pca_variance_threshold is not None and not apply_PCA_to_baseline(baseline):
                    continue

                X_train, X_test, y_test = apply_PCA_threshold(X_train_orig, X_test_orig, y_test_orig, pca_variance_threshold)
                clf = get_occ_from_name(baseline)
                
                for cutoff_type in [
                    'Empirical',
                    # 'Chi-squared',
                    'Bootstrap',
                    'Multisplit',
                ]:
                    if 'Multisplit' in cutoff_type and not apply_multisplit_to_baseline(baseline):
                        continue
                
                    if 'Bootstrap' in cutoff_type or 'Multisplit' in cutoff_type:
                        if 'Bootstrap' in cutoff_type:
                            resampling_method = 'Bootstrap'
                        else:
                            resampling_method = 'Multisplit'
                        resampling_threshold = prepare_resampling_threshold(clf, X_train, resampling_repeats, inlier_rate, method=resampling_method)

                    clf.fit(X_train)
                    scores = clf.score_samples(X_test)

                    if cutoff_type == 'Empirical':
                        emp_quantile = np.quantile(scores, q=1 - inlier_rate)
                        y_pred = np.where(scores > emp_quantile, 1, 0)
                    elif cutoff_type == 'Chi-squared':
                        d = X_test.shape[1]
                        chi_quantile = -scipy.stats.chi2.ppf(1 - inlier_rate, 2 * d)
                        y_pred = np.where(scores > chi_quantile, 1, 0)
                    elif 'Bootstrap' in cutoff_type or 'Multisplit' in cutoff_type:
                        y_pred = np.where(scores > resampling_threshold, 1, 0)
                    
                    test_metrics = get_metrics(y_test, y_pred, scores)

                    # print(f'{dataset}.{format}: {baseline}{f"+PCA{pca_variance_threshold:.1f}" if pca_variance_threshold is not None else ""} ({cutoff_type}, {exp+1}/{n_repeats})' + \
                    #     f' ||| AUC: {100 * auc:3.2f}, ACC: {100 * acc:3.2f}, F1: {100 * f1:3.2f}')
                    
                    occ_metrics = {
                        'Dataset': f'({format}) {dataset}',
                        'Method': baseline + (f"+PCA{pca_variance_threshold:.1f}" if pca_variance_threshold is not None else ""),
                        'Cutoff': cutoff_type,
                        'Exp': exp + 1,
                    }
                    for metric in metric_list:
                        if metric in occ_metrics and metric not in test_metrics:
                            continue
                        occ_metrics[metric] = test_metrics[metric]
                    
                    results.append(occ_metrics)
                    full_results.append(occ_metrics)
    
    df = pd.DataFrame.from_records(results)

    dataset_df = df[df.Dataset == f'({format}) {dataset}']
    res_df = dataset_df.groupby(['Dataset', 'Method', 'Cutoff'])\
        [metric_list] \
        .mean() \
        .round(4) \
        * 100

    res_df = append_mean_row(res_df)
    display(res_df)
    res_df.to_csv(os.path.join('results', f'dataset-{format}-{dataset}.csv'))

# Full result pivots
df = pd.DataFrame.from_records(full_results)
df

for metric in metric_list:
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
