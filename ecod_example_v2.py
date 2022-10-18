# %%
from ecod_v2 import ECODv2
from pyod.models.ecod import ECOD

import numpy as np

n = 100
N = 1000
outlier_proportion = 0.1

n_outliers = int(outlier_proportion * N)
n_inliers = N - n_outliers

X_train = np.random.multivariate_normal(np.zeros(2), np.eye(2), n)
X_test = np.concatenate([
    np.random.multivariate_normal(np.zeros(2), np.eye(2), n_inliers),
    np.random.multivariate_normal([5, 5], np.eye(2), n_outliers),
])
y_test = np.concatenate([np.zeros(n_inliers), np.ones(n_outliers)])

# %%
import matplotlib.pyplot as plt
from sklearn import metrics

clf = ECOD()
clf.fit(X_train)

scores = clf.decision_function(X_test)
scores
print(metrics.roc_auc_score(y_test, scores))

fig, axs = plt.subplots(1, 2, figsize=(16, 5))
scatter = axs[1].scatter(X_test[:, 0], X_test[:, 1], c=scores)
plt.colorbar(scatter, ax=axs[1])
axs[1].set_title('ECOD test scores (10% outliers)')

axs[0].scatter(X_train[:, 0], X_train[:, 1])
axs[0].set_xlim(*axs[1].get_xlim())
axs[0].set_ylim(*axs[1].get_ylim())
axs[0].set_title('Training set')

# plt.suptitle('ECOD - large test set')
plt.show()

# %%
import matplotlib.pyplot as plt

clf = ECODv2()
clf.fit(X_train)

scores = clf.decision_function(X_test)
scores
print(metrics.roc_auc_score(y_test, scores))

fig, axs = plt.subplots(1, 2, figsize=(16, 5))
scatter = axs[1].scatter(X_test[:, 0], X_test[:, 1], c=scores)
plt.colorbar(scatter, ax=axs[1])
axs[1].set_title('Modified ECOD (no train/test concat) test scores (10% outliers)')

axs[0].scatter(X_train[:, 0], X_train[:, 1])
axs[0].set_xlim(*axs[1].get_xlim())
axs[0].set_ylim(*axs[1].get_ylim())
axs[0].set_title('Training set')

# plt.suptitle('Modified ECOD (no train/test concat) - large test set')
plt.show()


# %%
