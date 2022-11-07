# %%
from occ_tests_common import *

import os
import occ_datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

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

metric_list = ['T1E']

for alpha in [0.05, 0.25, 0.5]:
    full_results = []

    RESULTS_DIR = f'results_t1e_{alpha:.2f}'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for (dataset, format) in datasets:
        print(f'({format}) {dataset} (alpha = {alpha:.2f})')
        results = []

        for exp in range(n_repeats):
            # Load data
            X, y = occ_datasets.load_dataset(dataset, format)
            X_train_orig, X_test_orig, y_test_orig = occ_datasets.split_occ_dataset(X, y, train_ratio=0.6)
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

                        # print(f'{dataset}.{format}: {baseline}{f"+PCA{pca_variance_threshold:.1f}" if pca_variance_threshold is not None else ""} ({cutoff_type}, {exp+1}/{n_repeats})' + \
                        #     f' ||| FNR: {fnr:.3f}')
                        occ_metrics = {
                            'Dataset': f'({format}) {dataset}',
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

        dataset_df = df[df.Dataset == f'({format}) {dataset}']
        res_df = dataset_df.groupby(['Dataset', 'Method', 'Cutoff', 'alpha'])\
            [metric_list] \
            .mean() \
            .round(3)

        res_df = append_mean_row(res_df)
        display(res_df)
        res_df.to_csv(os.path.join(RESULTS_DIR, f'dataset-{format}-{dataset}.csv'))

    # Full result pivots
    df = pd.DataFrame.from_records(full_results)
    df

    pivots = {}
    for metric in metric_list:
        metric_df = df
        
        pivot = metric_df \
            .pivot_table(values=metric, index=['Dataset'], columns=['Method', 'Cutoff'], dropna=False) \
            * 1
        
        pivots[metric] = pivot
        pivot = append_mean_row(pivot)
        
        pivot \
            .round(3) \
            .to_csv(os.path.join(RESULTS_DIR, f'dataset-all-{metric}.csv'))

    append_mean_row(pivots['T1E'] < alpha).to_csv(os.path.join(RESULTS_DIR, f'dataset-all-T1E-alpha.csv'))

# %%
