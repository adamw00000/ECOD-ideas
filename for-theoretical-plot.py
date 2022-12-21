# %%
import numpy as np
from occ_cutoffs import *
import scipy.stats
import scipy.optimize


alpha = 0.05

# theta = 0
pi = 0.1

def F(q):
    score = scipy.stats.norm.ppf(q)
    return scipy.stats.norm.cdf(-score - theta)

    # quantile = scipy.stats.norm.ppf(q)
    # return scipy.stats.norm.cdf(quantile, loc=theta, scale=1)

# def fun(u):
#     return np.abs((pi * u) / (pi * u + (1 - pi) * F(u)) - alpha * pi)

def left(u):
    return (pi * u) / (pi * u + (1 - pi) * F(u))

def right():
    return alpha

def fun(u):
    return np.abs(left(u) - right())

def for_fun(u):
    return ((1 - pi) * (1 - F(u))) / (pi * (1 - u) + (1 - pi) * (1 - F(u)))

thetas = []
us = []
fdrs = []
fors = []

for theta in np.linspace(0, 100, 100):
    res = scipy.optimize.differential_evolution(fun, bounds=[(0, 1)])
    u = res.x

    FDR = left(u)

    thetas.append(theta)
    us.append(u)
    fdrs.append(FDR)
    fors.append(for_fun(u))

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

plt.figure(figsize=(8, 6))
plt.plot(thetas, fors)
plt.plot(thetas, us)
plt.title(f'$\\pi$ = {pi}')

plt.show()

# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set_theme()

# pi = 0.6
# theta = 2

# plt.figure(figsize=(8, 6))
# x = np.linspace(0, 1, int(1e5) + 1)
# line = sns.lineplot(x=x, y=left(x))
# line.axhline(y = right(), xmin=0, xmax=1, c='r')
# plt.ylabel("Value")
# plt.xlabel("$u^*$")
# plt.legend(['Left side:$\\frac{\\pi u^*}{\\pi u^* + (1 - \\pi)F(u^*)}$', 'Right side: $\\pi \\alpha$'])
# plt.title(f'$\\theta$ = {theta}, $\\pi$ = {pi}, $\\alpha$ = {alpha}')
# # plt.savefig(f'plots/theta={str(theta).replace(".", "")}, pi={str(pi).replace(".", "")}.png', dpi=150)
# plt.show()

# %%
u = res.x
u

# %%
theta 

# %%
u = 0.99
(pi * u) / (pi * u + (1 - pi) * F(u))

# %%
alpha * pi

# %%
