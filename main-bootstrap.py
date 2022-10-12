# %%
import numpy as np

N = 100
center = 2
mean_p = [center, 0]
mean_n = [-center, 0]
std_p = std_n = [1, 1]
X = np.random.logistic(mean_p, std_p, (N, 2))
X_p = np.random.logistic(mean_p, std_p, (N, 2))
X_n = np.random.logistic(mean_n, std_n, (N, 2))
X_test = np.concatenate([X_p, X_n])

# %%
import matplotlib.pyplot as plt

plt.plot(X_p[:, 0], X_p[:, 1], 'r.')
plt.plot(X_n[:, 0], X_n[:, 1], 'b.')

# %%
from ecod_v2 import ECODv2

portion_positive = 0.5
thresholds = []

for _ in range(10):
    bootstrap_samples = np.random.choice(range(N), size=N, replace=True)
    is_bootstrap_sample = np.isin(range(N), bootstrap_samples)

    X_train, X_cal = X[is_bootstrap_sample], X[~is_bootstrap_sample]

    occ = ECODv2(n_jobs = -1, contamination=1e-4)
    occ.fit(X_train)
    
    scores = occ.decision_function(X_cal)
    empirical_quantile = np.quantile(scores, portion_positive)
    thresholds.append(empirical_quantile)

threshold = np.mean(thresholds)

# %%
from sklearn import metrics

y_pred = np.where(occ.decision_function(X_test) < threshold, 1, 0)
y_true = np.concatenate([np.ones(N), np.zeros(N)])

metrics.accuracy_score(y_true, y_pred)

# # %%
# y_pred = 1 - occ.predict(X_test)
# y_true = np.concatenate([np.ones(N), np.zeros(N)])

# metrics.accuracy_score(y_true, y_pred)

# %%
scores = occ.decision_function(X_test)
empirical_quantile = np.quantile(scores, portion_positive)
y_pred = np.where(scores < empirical_quantile, 1, 0)
y_true = np.concatenate([np.ones(N), np.zeros(N)])

metrics.accuracy_score(y_true, y_pred)

# %%
