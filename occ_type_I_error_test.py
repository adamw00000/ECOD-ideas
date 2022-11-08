from occ_all_tests_common import *

import os
import numpy as np
import pandas as pd
from IPython.display import display

n_repeats = 10
resampling_repeats = 10
metric_list = ['T1E']

def run_type_I_error_tests(DATASET_TYPE, get_all_distribution_configs):
    for alpha in [0.05, 0.25]:
        full_results = []

        RESULTS_DIR = f'results_{DATASET_TYPE}_t1e_{alpha:.2f}'
        os.makedirs(RESULTS_DIR, exist_ok=True)

        for (test_case_name, get_dataset_function) in get_all_distribution_configs():
            print(f'TYPE I ERROR tests: {test_case_name} (alpha = {alpha:.2f})')
            results = []

            for exp in range(n_repeats):
                np.random.seed(exp)
                # Load data
                X_train_orig, X_test_orig, y_test_orig = get_dataset_function()
                inlier_rate = np.mean(y_test_orig)

                # include only inliers
                X_test_orig, y_test_orig = filter_inliers(X_test_orig, y_test_orig)

                for baseline in [
                    # 'ECOD',
                    'ECODv2',
                    # 'ECODv2Min',
                    # 'GeomMedian',
                    'Mahalanobis',
                    # 'OC-SVM',
                    # 'IForest',
                ]:
                    for pca_variance_threshold in [None, 1.0]:
                        if pca_variance_threshold is not None and not apply_PCA_to_baseline(baseline):
                            continue

                        X_train, X_test, y_test = apply_PCA_threshold(X_train_orig, X_test_orig, y_test_orig, pca_variance_threshold)
                        clf = get_occ_from_name(baseline)
                        
                        for cutoff_type in [
                            'Empirical',
                            # 'Chi-squared',
                            # 'Bootstrap',
                            'Multisplit',
                            'Multisplit-1_repeat',
                            'Multisplit-1_median',
                        ]:
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
                                'alpha': alpha,
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
            res_df = dataset_df.groupby(['Dataset', 'Method', 'Cutoff', 'alpha'])\
                [metric_list] \
                .mean() \
                .round(3)

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
                .pivot_table(values=metric, index=['Dataset'], columns=['Method', 'Cutoff'], dropna=False) \
                * (100 if metric not in ['FDR', 'FRR', 'alpha', 'pi * alpha', '#', '#FD', '#D', 'T1E'] else 1)

            pivot = pivot.dropna(how='all')
            pivots[metric] = pivot
            pivot = append_mean_row(pivot)

            if metric in ['pi * alpha']:
                continue

            pivot \
                .round(3) \
                .to_csv(os.path.join(RESULTS_DIR, f'{DATASET_TYPE}-all-{metric}.csv'))

        append_mean_row(pivots['T1E'] < alpha).to_csv(os.path.join(RESULTS_DIR, f'{DATASET_TYPE}-all-T1E-alpha.csv'))
