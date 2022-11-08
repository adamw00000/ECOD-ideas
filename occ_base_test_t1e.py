from occ_all_tests_common import *

import os
import numpy as np
import pandas as pd
from IPython.display import display

n_repeats = 10
resampling_repeats = 10
metric_list = ['T1E']
alpha_metric = 'T1E'
metrics_to_multiply_by_100 = ['AUC', 'ACC', 'PRE', 'REC', 'F1']

test_description = 'Type I Error tests'

baselines = [
    # 'ECOD',
    'ECODv2',
    # 'ECODv2Min',
    # 'GeomMedian',
    'Mahalanobis',
    # 'OC-SVM',
    # 'IForest',
]
cutoffs = [
    'Empirical',
    # 'Chi-squared',
    # 'Bootstrap',
    'Multisplit',
    'Multisplit-1_repeat',
    'Multisplit-1_median',
]
pca_thresholds = [None, 1.0]

def run_type_I_error_tests(DATASET_TYPE, get_all_distribution_configs, alpha):
    full_results = []

    RESULTS_DIR = f'results_{DATASET_TYPE}_t1e_{alpha:.2f}'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for (test_case_name, get_dataset_function) in get_all_distribution_configs():
        print(f'{test_description}: {test_case_name} (alpha = {alpha:.2f})')
        results = []

        for exp in range(n_repeats):
            np.random.seed(exp)
            # Load data
            X_train_orig, X_test_orig, y_test_orig = get_dataset_function()
            inlier_rate = np.mean(y_test_orig)

            # include only inliers
            X_test_orig, y_test_orig = filter_inliers(X_test_orig, y_test_orig)

            for baseline in baselines:
                for pca_variance_threshold in pca_thresholds:
                    if pca_variance_threshold is not None and not apply_PCA_to_baseline(baseline):
                        continue

                    np.random.seed(exp)
                    X_train, X_test, y_test = apply_PCA_threshold(X_train_orig, X_test_orig, y_test_orig, pca_variance_threshold)
                    clf = get_occ_from_name(baseline)
                    
                    for cutoff_type in cutoffs:
                        if 'Multisplit' in cutoff_type and not apply_multisplit_to_baseline(baseline):
                            continue

                        np.random.seed(exp)
                        clf.fit(X_train)
                        scores = clf.score_samples(X_test)

                        occ_metrics = {
                            'Dataset': test_case_name,
                            'Method': baseline + (f"+PCA{pca_variance_threshold:.1f}" if pca_variance_threshold is not None else ""),
                            # 'Cutoff': cutoff_type,
                            'Exp': exp + 1,
                            '#': len(y_test),
                        }

                        if cutoff_type == 'Empirical':
                            emp_quantile = np.quantile(scores, q=1 - inlier_rate)
                            y_pred = np.where(scores > emp_quantile, 1, 0)
                        elif 'Multisplit' in cutoff_type:
                            np.random.seed(exp)
                            multisplit_cal_scores = prepare_multisplit_cal_scores(clf, X_train,
                                resampling_repeats=1 if '1_repeat' in cutoff_type else resampling_repeats)
                            
                            p_vals = get_multisplit_p_values(scores, multisplit_cal_scores,
                                median_multiplier=1 if '1_median' in cutoff_type else 2) # 2 should be correct
                            y_pred = np.where(p_vals < alpha, 0, 1)

                        occ_metrics['Cutoff'] = cutoff_type
                        method_metrics = prepare_metrics(y_test, y_pred, scores, occ_metrics, metric_list, pos_class_only=True)
                        results.append(method_metrics)
                        full_results.append(method_metrics)
        df = pd.DataFrame.from_records(results)

        dataset_df = df[df.Dataset == test_case_name]
        res_df = dataset_df.groupby(['Dataset', 'Method', 'Cutoff'])\
            [metric_list] \
            .mean()
        if alpha_metric is not None:
            res_df[f'{alpha_metric} < alpha'] = res_df[alpha_metric] < alpha

        res_df[metrics_to_multiply_by_100] = \
            (res_df[metrics_to_multiply_by_100] * 100).round(2)
        res_df[metric_list] = res_df[metric_list].round(3)

        res_df = append_mean_row(res_df)
        display(res_df)
        res_df.to_csv(os.path.join(RESULTS_DIR, f'{DATASET_TYPE}-{test_case_name}.csv'))

    # Full result pivots
    df = pd.DataFrame.from_records(full_results)
    df

    pivots = {}
    for metric in metric_list:
        metric_df = df
        if metric == 'AUC':
            metric_df = df.loc[df.Cutoff == 'Empirical']
        
        pivot = metric_df \
            .pivot_table(values=metric, index=['Dataset'], columns=['Method', 'Cutoff'], dropna=False)

        pivot = pivot.dropna(how='all')
        pivots[metric] = pivot
        pivot = append_mean_row(pivot)

        if metric in metrics_to_multiply_by_100:
            pivot = (pivot * 100).round(2)
        else:
            pivot = pivot.round(3)

        pivot \
            .to_csv(os.path.join(RESULTS_DIR, f'{DATASET_TYPE}-all-{metric}.csv'))

    if alpha_metric is not None:
        append_mean_row(pivots[alpha_metric] < alpha).to_csv(os.path.join(RESULTS_DIR, f'{DATASET_TYPE}-all-{alpha_metric}-alpha.csv'))
