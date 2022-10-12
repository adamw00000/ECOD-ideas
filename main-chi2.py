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

# zależność
#   autoregresja rzędu 1: sigma = ij => rho (-1; 1) ^ |i-j|
#   Normalny(0, I) * sqrt(sigma)
#   scatterplot: odkręcona elipsa

#   wykładniczy wielowymiarowy
# liczba wymiarów ~10
# siatka

# %%
import matplotlib.pyplot as plt

plt.plot(X[:, 0], X[:, 1], 'g.')

# %%
import matplotlib.pyplot as plt

plt.plot(X_p[:, 0], X_p[:, 1], 'r.')
plt.plot(X_n[:, 0], X_n[:, 1], 'b.')

# %%
from ecod_v2 import ECODv2

occ = ECODv2(n_jobs = -1, contamination=1e-4)
occ.fit(X)

# %%
portion_positive = 0.5

d = 2
import scipy.stats
chi_quantile = scipy.stats.chi2.ppf(portion_positive, 2 * d)
chi_quantile

# %%
from sklearn import metrics

y_pred = np.where(occ.decision_function(X_test) < chi_quantile, 1, 0)
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
