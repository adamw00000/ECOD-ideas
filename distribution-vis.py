# %%
import os
from occ_tests_common import *
import numpy as np
import matplotlib.pyplot as plt


n=1_000
dim=2
rho=0.5

os.makedirs('plots', exist_ok=True)

plt.figure(figsize=(8, 6))
auto_reg_train, auto_reg_test, _ = sample_autoregressive(n, dim, rho=0.5)
plt.plot(auto_reg_train[:, 0], auto_reg_train[:, 1], '.')
plt.plot(auto_reg_test[:, 0], auto_reg_test[:, 1], '.')
plt.title('Autoregressive dataset')
plt.legend(['Train', 'Test'])
plt.savefig(os.path.join('plots', 'autoregressive.png'), dpi=300)

# %%
plt.figure(figsize=(8, 6))
multinomial_train, multinomial_test, _ = sample_exponential(n, dim)
plt.plot(multinomial_train[:, 0], multinomial_train[:, 1], '.')
plt.plot(multinomial_test[:, 0], multinomial_test[:, 1], '.')
plt.title('Multinomial dataset')
plt.legend(['Train', 'Test'])
plt.savefig(os.path.join('plots', 'multinomial.png'), dpi=300)

# %%
plt.figure(figsize=(8, 6))
normal_train, normal_test, _ = sample_normal(n, dim)
plt.plot(normal_train[:, 0], normal_train[:, 1], '.')
plt.plot(normal_test[:, 0], normal_test[:, 1], '.')
plt.title('Normal dataset')
plt.legend(['Train', 'Test'])
plt.savefig(os.path.join('plots', 'normal.png'), dpi=300)

# %%
