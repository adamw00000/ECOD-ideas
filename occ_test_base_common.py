from occ_all_tests_common import *
from occ_cutoffs import *

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from typing import List

n_repeats = 10
resampling_repeats = 10

def run_tests(metric_list, alpha_metrics, test_description, get_results_dir, baselines, get_cutoffs, pca_thresholds,
        DATASET_TYPE, get_all_distribution_configs, alpha, test_inliers_only=False, visualize_tests=True, apply_control_cutoffs=True):
    full_results = []

    RESULTS_DIR = get_results_dir(DATASET_TYPE, alpha)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for (test_case_name, get_dataset_function) in get_all_distribution_configs():
        print(f'{test_description}: {test_case_name}' + \
            f' (alpha = {alpha:.2f})' if alpha is not None else '')
        results = []

        for exp in range(n_repeats):
            np.random.seed(exp)
            # Load data
            X_train_orig, X_test_orig, y_test_orig = get_dataset_function()
            inlier_rate = np.mean(y_test_orig)

            if test_inliers_only:
                # include only inliers
                X_test_orig, y_test_orig = filter_inliers(X_test_orig, y_test_orig)

            for clf_name in baselines:
                for pca_variance_threshold in pca_thresholds:
                    if pca_variance_threshold is not None and not apply_PCA_to_baseline(clf_name):
                        continue

                    np.random.seed(exp)
                    X_train, X_test, y_test = apply_PCA_threshold(X_train_orig, X_test_orig, y_test_orig, pca_variance_threshold)
                    construct_clf = lambda clf_name=clf_name, exp=exp, RESULTS_DIR=RESULTS_DIR: \
                        get_occ_from_name(clf_name, random_state=exp, RESULTS_DIR=RESULTS_DIR)

                    extra_params = {
                        'control_cutoff_params': {
                            'alpha': alpha,
                            'inlier_rate': inlier_rate,
                        },
                        'common_visualization_params': {
                            'test_case_name': test_case_name,
                            'clf_name': clf_name,
                            'RESULTS_DIR': RESULTS_DIR,
                        },
                        'special_visualization_params': {
                            'exp': exp,
                            'pca_variance_threshold': pca_variance_threshold,
                            'X_train': X_train,
                            'X_test': X_test,
                            'y_test': y_test,
                        },
                    }

                    for cutoff in get_cutoffs(construct_clf, alpha, resampling_repeats):
                        if not apply_multisplit_to_baseline(clf_name) and (isinstance(cutoff, MultisplitCutoff) or isinstance(cutoff, MultisplitThresholdCutoff) or isinstance(cutoff, NoSplitCutoff)):
                            continue

                        np.random.seed(exp)
                        predictions = get_cutoff_predictions(cutoff, X_train, X_test, inlier_rate, 
                            visualize_tests, apply_control_cutoffs, **extra_params)

                        for cutoff_name, scores, y_pred, elapsed in predictions:
                            occ_metrics = {
                                'Dataset': test_case_name,
                                'Method': clf_name + (f"+PCA{pca_variance_threshold:.1f}" if pca_variance_threshold is not None else ""),
                                'Cutoff': cutoff_name,
                                'Exp': exp + 1,
                                '#': len(y_test),
                                'Time': elapsed,
                            }

                            method_metrics = prepare_metrics(y_test, y_pred, scores, occ_metrics, metric_list, pos_class_only=test_inliers_only)
                            results.append(method_metrics)
                            full_results.append(method_metrics)

        df = pd.DataFrame.from_records(results)

        dataset_df = df[df.Dataset == test_case_name]
        res_df = dataset_df.groupby(['Dataset', 'Method', 'Cutoff'])\
            [metric_list] \
            .mean()
        for alpha_metric in alpha_metrics:
            res_df[f'{alpha_metric} < alpha'] = res_df[alpha_metric] < alpha

        for metric in metric_list:
            res_df[metric] = round_and_multiply_metric(res_df[metric], metric)

        res_df = append_mean_row(res_df)
        display(res_df)
        res_df.to_csv(os.path.join(RESULTS_DIR, f'{DATASET_TYPE}-{test_case_name}.csv'))

    # Full result pivots
    df = pd.DataFrame.from_records(full_results)
    df

    pivots = {}
    for metric in metric_list:
        metric_df = df
        # if metric == 'AUC':
        #     metric_df = df.loc[np.isin(df.Cutoff, ['Empirical'])]
        
        pivot = metric_df \
            .pivot_table(values=metric, index=['Dataset'], columns=['Method', 'Cutoff'], dropna=False)

        pivot = pivot.dropna(how='all')
        pivots[metric] = pivot
        pivot = append_mean_row(pivot)
        pivot = round_and_multiply_metric(pivot, metric)

        pivot \
            .to_csv(os.path.join(RESULTS_DIR, f'{DATASET_TYPE}-all-{metric}.csv'))

    for alpha_metric in alpha_metrics:
        append_mean_row(pivots[alpha_metric] < alpha).to_csv(os.path.join(RESULTS_DIR, f'{DATASET_TYPE}-all-{alpha_metric}-alpha.csv'))
