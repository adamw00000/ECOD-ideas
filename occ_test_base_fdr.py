from occ_all_tests_common import *
from occ_cutoffs import *
from occ_test_base_common import run_tests
from typing import List

metric_list = ['AUC', 'ACC', 'PRE', 'REC', 'F1', 
    'T1E', 'FOR', 'FNR', '#', '#FD', '#D', 'FDR', 'Time']
alpha_metrics = ['FDR', 'FOR', 'FNR']

test_description = 'FDR tests'
get_results_dir = lambda dataset_type, alpha: f'results_{dataset_type}_fdr_{alpha:.2f}'

baselines = [
    'A^3',
    # 'ECOD',
    'ECODv2',
    # 'ECODv2Min',
    # 'GeomMedian',
    'Mahalanobis',
    # 'OC-SVM',
    'IForest',
    'HBOS',
    'CBLOF',
]
def get_cutoffs(construct_clf, alpha, resampling_repeats) -> List[Cutoff]:
    return [
        EmpiricalCutoff(construct_clf),
        # ChiSquaredCutoff(construct_clf, dim),
        # BootstrapThresholdCutoff(construct_clf, resampling_repeats),
        # MultisplitThresholdCutoff(construct_clf, resampling_repeats),
        # MultisplitCutoff(construct_clf, alpha, resampling_repeats=resampling_repeats, median_multiplier=2),
        # MultisplitCutoff(construct_clf, alpha, resampling_repeats=1, median_multiplier=2),
        MultisplitCutoff(construct_clf, alpha, resampling_repeats=resampling_repeats, median_multiplier=1),
        NoSplitCutoff(construct_clf, alpha),
    ]
pca_thresholds = [None, 1.0]

def run_fdr_tests(DATASET_TYPE, get_all_distribution_configs, alpha):
    run_tests(metric_list, alpha_metrics, test_description, get_results_dir, baselines, get_cutoffs, pca_thresholds,
        DATASET_TYPE, get_all_distribution_configs, alpha, test_inliers_only=False, visualize_tests=True, apply_control_cutoffs=True)