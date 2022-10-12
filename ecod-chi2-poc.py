# %%
import numpy as np

n = 100000
d = 10

X1 = np.random.randn(n)
X2 = np.random.exponential(1, n)
X1, X2
Xs = [np.random.randn(n) for _ in range(d)]

# %%
from statsmodels.distributions.empirical_distribution import ECDF

ecdf1 = ECDF(X1)
ecdf2 = ECDF(X2)
ecdfs = [ECDF(X) for X in Xs]

# %%
import matplotlib.pyplot as plt

plt.hist(ecdf1(X1))
plt.show()
plt.hist(ecdf2(X2))
plt.show()

# %%
plt.hist(-(np.log(ecdf1(X1))))

# %%
sum = np.sum([-(np.log(ecdf(X))) for ecdf, X in zip(ecdfs, Xs)], axis=0)
plt.hist(sum)

# %%
import scipy.stats
chi = scipy.stats.chi2.rvs(2 * d, size=n)
# X3 = np.random.chisquare(1, n)
plt.hist(chi)

# %%
scipy.stats.chi2.rvs(2 * d, size=n)