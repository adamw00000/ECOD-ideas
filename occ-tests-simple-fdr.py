# %%
from occ_all_tests_lib import *
from occ_test_base_fdr import run_fdr_tests

DATASET_TYPE = 'SIMPLEtest'

def sample_normal_simple(n, diff=2, inlier_rate=0.6):
    n_inliers = int(inlier_rate * n)
    n_outliers = n - n_inliers

    X_train = np.random.multivariate_normal(np.zeros(2), np.eye(2), n)
    X_test = np.concatenate([
        np.random.multivariate_normal([0, 0], np.eye(2), n_inliers),
        np.random.multivariate_normal([diff, 0], np.eye(2), n_outliers),
    ])
    y_test = np.concatenate([np.ones(n_inliers), np.zeros(n_outliers)])
    return X_train, X_test, y_test

def get_all_distribution_configs():
    all_configs = []

    for distribution, get_data in [
        ('SimpleNormal', sample_normal_simple),
    ]:
        for num_samples in [1_000]:
            test_case_name = f'{distribution} ({num_samples})'
            get_dataset_function = \
                lambda get_data=get_data, num_samples=num_samples: \
                    get_data(num_samples)

            all_configs.append((test_case_name, get_dataset_function))
    
    return all_configs

for alpha in [0.1, 0.25]:
    run_fdr_tests(DATASET_TYPE, get_all_distribution_configs, alpha=alpha)

# %%
