# %%
from occ_tests_common import *

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

n_repeats = 10
resampling_repeats = 10

metric_list = ['T1E']

for alpha in [0.05, 0.25, 0.5]:
    full_results = []

    RESULTS_DIR = f'resultsdist_t1e_{alpha:.2f}'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for distribution, get_data in [
        ('Normal', sample_normal),
        ('Exponential', sample_exponential),
        ('Autoregressive', sample_autoregressive)
    ]:
        results = []

        for num_samples in [100, 10_000]:
            for dim in [2, 10, 50]:
                if num_samples == 100 and dim == 50:
                    continue

                print(f'{distribution} ({num_samples}x{dim}, alpha = {alpha:.2f})')
                for exp in range(n_repeats):
                    # Load data
                    X_train_orig, X_test_orig, y_test_orig = get_data(num_samples, dim)
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
                                'Multisplit',
                            ]:
                                if 'Multisplit' in cutoff_type and not apply_multisplit_to_baseline(baseline):
                                    continue
                                
                                if 'Multisplit' in cutoff_type:
                                    multisplit_cal_scores = prepare_multisplit_cal_scores(clf, X_train, resampling_repeats)

                                clf.fit(X_train)
                                scores = clf.score_samples(X_test)

                                if 'Multisplit' in cutoff_type:
                                    p_vals = get_multisplit_p_values(scores, multisplit_cal_scores, median_multiplier=2)
                                    y_pred = np.where(p_vals < alpha, 0, 1)
                            
                                test_metrics = get_metrics(y_test, y_pred, scores, one_class_only=True)

                                # print(f'{distribution} ({num_samples}x{dim}): {baseline}{f"+PCA{pca_variance_threshold:.1f}" if pca_variance_threshold is not None else ""} ({cutoff_type}, {exp+1}/{n_repeats})' + \
                                #     f' ||| FNR: {fnr:.3f}')
                                occ_metrics = {
                                    'Distribution': distribution,
                                    'N': num_samples,
                                    'Dim': dim,
                                    'Method': baseline + (f"+PCA{pca_variance_threshold:.1f}" if pca_variance_threshold is not None else ""),
                                    'Cutoff': cutoff_type,
                                    'Exp': exp + 1,
                                    'alpha': alpha,
                                }
                                for metric in metric_list:
                                    if metric in occ_metrics and metric not in test_metrics:
                                        continue
                                    occ_metrics[metric] = test_metrics[metric]
                                
                                results.append(occ_metrics)
                                full_results.append(occ_metrics)
                
        df = pd.DataFrame.from_records(results)

        dist_df = df[df.Distribution == distribution]
        res_df = dist_df.groupby(['Distribution', 'N', 'Dim', 'Method', 'Cutoff', 'alpha'])\
            [metric_list] \
            .mean() \
            .round(3)
        
        res_df = append_mean_row(res_df)
        display(res_df)
        res_df.to_csv(os.path.join(RESULTS_DIR, f'dist-{distribution}.csv'))

    # Full result pivots
    df = pd.DataFrame.from_records(full_results)
    df

    pivots = {}
    for metric in metric_list:
        metric_df = df
        
        pivot = metric_df \
            .pivot_table(values=metric, index=['Distribution', 'N', 'Dim'], columns=['Method', 'Cutoff'], dropna=False) \
            * 1
        
        pivot = pivot.dropna(how='all')
        pivots[metric] = pivot
        pivot = append_mean_row(pivot)
        
        pivot \
            .round(3) \
            .to_csv(os.path.join(RESULTS_DIR, f'dist-all-{metric}.csv'))

    append_mean_row(pivots['T1E'] < alpha).to_csv(os.path.join(RESULTS_DIR, f'dist-all-T1E-alpha.csv'))

# %%
