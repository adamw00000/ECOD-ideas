# %%
import numpy as np
from occ_cutoffs import *
import scipy.stats
import scipy.optimize

import matplotlib.pyplot as plt
import seaborn as sns

def F(u):
    score = scipy.stats.norm.ppf(u)
    # return 1 - scipy.stats.norm.cdf(-score - theta) # Assume IN ~ N(0, I), OUT ~ N(theta, I)
    return scipy.stats.norm.cdf(score + theta) # Assume IN ~ N(0, I), OUT ~ N(-theta, I)

def left(u):
    return (pi * u) / (pi * u + (1 - pi) * F(u))

def fun(u, threshold):
    return np.abs(left(u) - threshold)

def for_fun(u):
    return ((1 - pi) * (1 - F(u))) / (pi * (1 - u) + (1 - pi) * (1 - F(u)))

alpha = 0.05

for threshold in ['BH', 'BH+pi']:
    sns.set_theme()
    plt.figure(figsize=(10, 8))

    for pi in np.linspace(0, 1, 11):
        thetas = []
        us = []
        fdrs = []
        fors = []

        for theta in np.linspace(0, 5, 21):
            if threshold == 'BH':
                threshold_value = alpha * pi
            elif threshold == 'BH+pi':
                threshold_value = alpha

            res = scipy.optimize.differential_evolution(fun, 
                bounds=[(0, 1)], tol=5e-4, args=(threshold_value,)
            )
            u = res.x

            FDR = left(u)

            thetas.append(theta)
            us.append(u)
            fdrs.append(FDR)
            fors.append(for_fun(u).item())

        sns.lineplot(x=thetas, y=fors, label=f'$\\pi = {pi:.1f}$')

    plt.xlabel('Distance between means $\\theta$')
    plt.ylabel('Expected FOR')

    # plt.title(f'Expected FOR value, FDR control ({threshold})')
    plt.legend()
    plt.savefig(os.path.join('plots', f'FOR_theoretical_{threshold}.png'),
        bbox_inches='tight', facecolor='w', dpi=600)
    plt.savefig(os.path.join('plots', f'FOR_theoretical_{threshold}.pdf'),
        bbox_inches='tight', facecolor='w')
    plt.show()

# %%
