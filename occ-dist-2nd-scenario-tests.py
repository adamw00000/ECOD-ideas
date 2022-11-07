# %%
from occ_tests_common import *

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

n_repeats = 10
resampling_repeats = 10

metric_list = ['pi * alpha', 'AUC', 'ACC', 'PRE', 'REC', 'F1', '#', '#FD', '#D', 'FDR']

for alpha in [0.05, 0.25, 0.5]:
    full_results = []

    RESULTS_DIR = f'resultsdist_fdr_{alpha:.2f}'
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
                                'Multisplit+BH',
                                'Multisplit+BH+pi',
                            ]:
                                if 'Multisplit' in cutoff_type and not apply_multisplit_to_baseline(baseline):
                                    continue

                                if 'Multisplit' in cutoff_type:
                                    multisplit_cal_scores = prepare_multisplit_cal_scores(clf, X_train, resampling_repeats)

                                clf.fit(X_train)
                                scores = clf.score_samples(X_test)

                                if cutoff_type == 'Empirical':
                                    emp_quantile = np.quantile(scores, q=1 - inlier_rate)
                                    y_pred = np.where(scores > emp_quantile, 1, 0)
                                elif 'Multisplit' in cutoff_type:
                                    p_vals = get_multisplit_p_values(scores, multisplit_cal_scores, median_multiplier=2)
                                    y_pred = np.where(p_vals < alpha, 0, 1)

                                    if 'BH' in cutoff_type:
                                        if 'pi' in cutoff_type:
                                            y_pred = use_BH_procedure(p_vals, alpha, pi=inlier_rate)
                                        else:
                                            y_pred = use_BH_procedure(p_vals, alpha, pi=None)

                                test_metrics = get_metrics(y_test, y_pred, scores)

                                # print(f'{distribution} ({num_samples}x{dim}): {baseline}{f"+PCA{pca_variance_threshold:.1f}" if pca_variance_threshold is not None else ""} ({cutoff_type}, {exp+1}/{n_repeats})' + \
                                #     f' ||| AUC: {100 * auc:3.2f}, ACC: {100 * acc:3.2f}, F1: {100 * f1:3.2f}, FDR: {fdr:.3f}')
                                occ_metrics = {
                                    'Distribution': distribution,
                                    'N': num_samples,
                                    'Dim': dim,
                                    'Method': baseline + (f"+PCA{pca_variance_threshold:.1f}" if pca_variance_threshold is not None else ""),
                                    'Cutoff': cutoff_type,
                                    'Exp': exp + 1,
                                    'alpha': alpha,
                                    'pi * alpha': inlier_rate * alpha,
                                    '#': len(y_pred),
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
        res_df['FDR < alpha'] = res_df['FDR'] < alpha
        res_df['FDR < pi * alpha'] = (res_df['FDR'] < res_df['pi * alpha'])

        res_df = append_mean_row(res_df)
        
        display(res_df)
        res_df.to_csv(os.path.join(RESULTS_DIR, f'dist-{distribution}.csv'))

    # Full result pivots
    df = pd.DataFrame.from_records(full_results)
    df

    pivots = {}
    for metric in metric_list:
        metric_df = df
        if metric == 'AUC':
            metric_df = df.loc[df.Cutoff == 'Empirical']
        
        pivot = metric_df \
            .pivot_table(values=metric, index=['Distribution', 'N', 'Dim'], columns=['Method', 'Cutoff'], dropna=False) \
            * (100 if metric not in ['FDR', 'alpha', 'pi * alpha'] else 1)

        pivot = pivot.dropna(how='all')
        pivots[metric] = pivot
        pivot = append_mean_row(pivot)

        if metric in ['pi * alpha']:
            continue

        pivot \
            .applymap("{0:.2f}".format if metric != 'FDR' else "{0:.3f}".format ) \
            .to_csv(os.path.join(RESULTS_DIR, f'dist-all-{metric}.csv'))

    append_mean_row(pivots['FDR'] < alpha).to_csv(os.path.join(RESULTS_DIR, f'dist-all-FDR-alpha.csv'))
    append_mean_row(pivots['FDR'] < pivots['pi * alpha']).to_csv(os.path.join(RESULTS_DIR, f'dist-all-FDR-pi-alpha.csv'))

# %%
