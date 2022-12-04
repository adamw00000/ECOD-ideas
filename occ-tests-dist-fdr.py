# %%
from occ_all_tests_lib import *
from occ_test_base_fdr import run_fdr_tests

DATASET_TYPE = 'GENdist'

def get_all_distribution_configs():
    all_configs = []

    for distribution, get_data in [
        ('Normal', sample_normal),
        ('Exponential', sample_exponential),
        ('Autoregressive', sample_autoregressive)
    ]:
        for num_samples in [100, 10_000]:
            for dim in [2, 20]:                    
                test_case_name = f'{distribution} ({num_samples}x{dim})'
                get_dataset_function = \
                    lambda get_data=get_data, num_samples=num_samples, dim=dim: \
                        get_data(num_samples, dim)

                all_configs.append((test_case_name, get_dataset_function))
    
    return all_configs

for alpha in [0.1, 0.25]:
    run_fdr_tests(DATASET_TYPE, get_all_distribution_configs, alpha=alpha)

# %%
