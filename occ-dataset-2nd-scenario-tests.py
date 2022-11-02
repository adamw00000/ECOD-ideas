# %%
from occ_tests_common import *

import os
import occ_datasets
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from IPython.display import display

from pyod.models.ecod import ECOD
from ecod_v2 import ECODv2
from ecod_v2_min import ECODv2Min
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

n_repeats = 10
resampling_repeats = 10

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

# datasets = [
#     ('Hepatitis', 'arff'),
# ]

for alpha in [0.05, 0.25, 0.5]:
    full_results = []

    RESULTS_DIR = f'results_fdr_{alpha:.2f}'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    for (dataset, format) in datasets:
        results = []

        for exp in range(n_repeats):
            # Load data
            X, y = occ_datasets.load_dataset(dataset, format)
            X_train_orig, X_test_orig, y_test_orig = occ_datasets.split_occ_dataset(X, y, train_ratio=0.6)
            inlier_rate = np.mean(y_test_orig)

            for baseline in [
                # 'ECOD',
                'ECODv2',
                # 'ECODv2Min',
                # 'GeomMedian',
                'Mahalanobis',
                # 'OC-SVM',
                # 'IForest',
            ]:
                X_train, X_test, y_test = X_train_orig, X_test_orig, y_test_orig
                for pca_variance_threshold in [0.5, 0.9, 1.0, None]:
                # for pca_variance_threshold in [None]:
                    if pca_variance_threshold is not None:
                        if not 'ECODv2' in baseline and not 'Mahalanobis' in baseline:
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
                        # 'Chi-squared',
                        # 'Bootstrap',
                        'Multisplit',
                        'Multisplit+BH',
                        'Multisplit+BH+pi',
                    ]:
                        if cutoff_type != 'Empirical' and not 'ECODv2' in baseline and not 'Mahalanobis' in baseline:
                            continue
                        
                        N = len(X_train)
                        if cutoff_type == 'Multisplit':
                            cal_scores_all = np.zeros((resampling_repeats, N - int(N/2)))

                            for i in range(resampling_repeats):
                                resampling_samples = np.random.choice(range(N), size=int(N/2), replace=False)
                                is_selected_sample = np.isin(range(N), resampling_samples)
                                X_resampling_train, X_resampling_cal = X_train[is_selected_sample], X_train[~is_selected_sample]
                                
                                clf.fit(X_resampling_train)
                                cal_scores = clf.score_samples(X_resampling_cal)
                                cal_scores_all[i, :] = cal_scores

                        clf.fit(X_train)

                        scores = clf.score_samples(X_test)
                        auc = metrics.roc_auc_score(y_test, scores)


                        if cutoff_type == 'Empirical':
                            emp_quantile = np.quantile(scores, q=1 - inlier_rate)
                            y_pred = np.where(scores > emp_quantile, 1, 0)
                        elif 'Multisplit' in cutoff_type:
                            p_vals_all = np.zeros((resampling_repeats, len(scores)))
                            for i in range(resampling_repeats):
                                cal_scores = cal_scores_all[i, :]
                                num_smaller_cal_scores = (scores > cal_scores.reshape(-1, 1)).sum(axis=0)
                                p_vals = (num_smaller_cal_scores + 1) / (len(cal_scores) + 1)
                                p_vals_all[i, :] = p_vals
                            p_vals = 2 * np.median(p_vals_all, axis=0)
                            y_pred = np.where(p_vals < alpha, 0, 1)

                            if 'BH' in cutoff_type:
                                fdr_ctl_threshold = alpha
                                if 'pi' in cutoff_type:
                                    pi = inlier_rate
                                    fdr_ctl_threshold = alpha / pi
                                
                                sorted_indices = np.argsort(p_vals)
                                bh_thresholds = np.linspace(
                                    fdr_ctl_threshold / len(p_vals), fdr_ctl_threshold, len(p_vals))
                                
                                # is_h0_rejected == is_outlier, H_0: X ~ P_X
                                # OLD WAY
                                is_h0_rejected = p_vals[sorted_indices] < bh_thresholds
                                # NEW WAY
                                rejections = np.where(is_h0_rejected)[0]
                                if len(rejections) > 0:
                                    is_h0_rejected[:(np.max(rejections) + 1)] = True
                                # take all the point to the left of last discovery

                                y_pred = np.ones_like(p_vals)
                                y_pred[sorted_indices[is_h0_rejected]] = 0

                        false_detections = np.sum((y_pred == 0) & (y_test == 1))
                        detections = np.sum(y_pred == 0)
                        fdr = false_detections / detections

                        acc = metrics.accuracy_score(y_test, y_pred)
                        pre = metrics.precision_score(y_test, y_pred)
                        rec = metrics.recall_score(y_test, y_pred)
                        f1 = metrics.f1_score(y_test, y_pred)

                        print(f'{dataset}.{format}: {baseline}{f"+PCA{pca_variance_threshold:.1f}" if pca_variance_threshold is not None else ""} ({cutoff_type}, {exp+1}/{n_repeats})' + \
                            f' ||| AUC: {100 * auc:3.2f}, ACC: {100 * acc:3.2f}, F1: {100 * f1:3.2f}, FDR: {fdr:.3f}')
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
                            'alpha': alpha,
                            'pi * alpha': alpha * inlier_rate,
                            '#': len(y_pred),
                            '#FD': false_detections,
                            '#D': detections,
                        }
                        results.append(occ_metrics)
                        full_results.append(occ_metrics)
        
        df = pd.DataFrame.from_records(results)

        dataset_df = df[df.Dataset == f'({format}) {dataset}']
        res_df = dataset_df.groupby(['Dataset', 'Method', 'Cutoff', 'alpha'])\
            [['pi * alpha', 'AUC', 'ACC', 'PRE', 'REC', 'F1', '#', '#FD', '#D', 'FDR']] \
            .mean()

        res_df[['AUC', 'ACC', 'PRE', 'REC', 'F1']] = (res_df[['AUC', 'ACC', 'PRE', 'REC', 'F1']] * 100) \
            .applymap('{0:.2f}'.format)
        res_df[['#FD', '#D']] = (res_df[['#FD', '#D']]) \
            .applymap('{0:.1f}'.format)
        res_df['FDR < alpha'] = res_df['FDR'] < res_df.index.get_level_values('alpha')
        res_df['FDR < pi * alpha'] = (res_df['FDR'] < res_df['pi * alpha'])
        res_df[['FDR', 'pi * alpha']] = res_df[['FDR', 'pi * alpha']].applymap('{0:.3f}'.format)

        res_df = append_mean_row(res_df)
        display(res_df)
        res_df.to_csv(os.path.join(RESULTS_DIR, f'dataset-{format}-{dataset}.csv'))

    # Full result pivots
    df = pd.DataFrame.from_records(full_results)
    df

    pivots = {}
    for metric in ['AUC', 'ACC', 'F1', 'PRE', 'REC', 'FDR', 'alpha', 'pi * alpha']:
        metric_df = df
        if metric == 'AUC':
            metric_df = df.loc[df.Cutoff == 'Empirical']
        
        pivot = metric_df \
            .pivot_table(values=metric, index=['Dataset'], columns=['Method', 'Cutoff'], dropna=False) \
            * (100 if metric not in ['FDR', 'alpha', 'pi * alpha'] else 1)

        pivots[metric] = pivot
        pivot = append_mean_row(pivot)

        if metric in ['alpha', 'pi * alpha']:
            continue

        pivot \
            .applymap("{0:.2f}".format if metric != 'FDR' else "{0:.3f}".format ) \
            .to_csv(os.path.join(RESULTS_DIR, f'dataset-all-{metric}.csv'))

    append_mean_row(pivots['FDR'] < pivots['alpha']).to_csv(os.path.join(RESULTS_DIR, f'dataset-all-FDR-alpha.csv'))
    append_mean_row(pivots['FDR'] < pivots['pi * alpha']).to_csv(os.path.join(RESULTS_DIR, f'dataset-all-FDR-pi-alpha.csv'))

# %%
