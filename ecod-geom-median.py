# %%
import numpy as np

n = 1000
d = 10

X1 = np.random.randn(n)
X2 = np.random.exponential(1, n)
X1, X2
Xs = [np.random.randn(n) for _ in range(d)]

# %%
import numpy as np
from scipy.spatial.distance import cdist, euclidean

def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1

# %%
X = np.concatenate([X1.reshape(-1, 1), X2.reshape(-1, 1)], axis=1)
med = geometric_median(np.concatenate([X1.reshape(-1, 1), X2.reshape(-1, 1)], axis=1))
med

# %%
import matplotlib.pyplot as plt
import seaborn as sns
# plt.plot(X1, X2, '.')
# plt.plot(med[0], med[1], 'ro')


def tp(c, X, med):
    def s(X):
        return (X - c) * (med - c)
    
    Y = X - (med - c)
    
    print((np.sum(np.all(s(Y) < 0, axis=1)) + np.sum(np.all(s(Y) == 0, axis=1)) / 2) / len(Y))
    return (np.sum(np.all(s(X) < 0, axis=1)) + np.sum(np.all(s(X) == 0, axis=1)) / 2) / \
           (np.sum(np.all(s(Y) < 0, axis=1)) + np.sum(np.all(s(Y) == 0, axis=1)) / 2)
    

tps = [tp(x, X, med) for x in X]
tps

# plt.plot(X1, X2, '.', c=tps)
sns.scatterplot(X1, X2, c=tps)
plt.plot(med[0], med[1], 'ro')

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