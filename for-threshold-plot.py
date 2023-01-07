# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def G(u):
    return 1 - (1 - u) ** 2

def left_side(u):
    return 1 - G(u)

def right_side(u):
    return pi / (1 - alpha) * (1 - u)

pi = 0.3
alpha = 0.05

u = np.linspace(0, 1, 1000)

plt.figure(figsize=(10, 8))

sns.lineplot(x=u, y=left_side(u), label='$1 - G(u)$')
sns.lineplot(x=u, y=right_side(u), label='$\\frac{\pi}{1 - \\alpha} \ (1 - u)$')
