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
import math

portion_positive = 0.5
thresholds = []

subsets = []
L = 3
Q = N
k = int(math.ceil(Q ** 0.4))

for l in range(L):
    bootstrap_samples = np.random.choice(range(N), size=Q, replace=True)
    is_bootstrap_sample = np.isin(range(N), bootstrap_samples)

    X_sel = X[is_bootstrap_sample]
    subsets.append(X_sel)

from sklearn.neighbors import NearestNeighbors

training_distances = []

# Train
def kth_neighbor_distance(search_set, query, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    nbrs = nbrs.fit(search_set)
    distances, _ = nbrs.kneighbors(query)

    kth_neighbor_dist = distances[:, k - 1]
    return kth_neighbor_dist

for query in range(L):
    X_query = subsets[query]
    knn_distances = np.zeros(len(X_query))

    for search_set in range(L):
        if query == search_set:
            continue
        X_search = subsets[search_set]

        kth_neighbor_dist = kth_neighbor_distance(X_search, X_query, k)
        knn_distances += kth_neighbor_dist / (L - 1) # mean
    
    training_distances.append(knn_distances)

# Predict
scores_full = np.zeros((len(X_test), L))

for search_set in range(L):
    X_search = subsets[search_set]
    kth_neighbor_dist = kth_neighbor_distance(X_search, X_test, k)
    
    train_dist = training_distances[search_set]

    # Loop
    scores = np.zeros_like(kth_neighbor_dist)
    for q in range(len(X_test)):
        scores[q] = np.mean(kth_neighbor_dist[q] < train_dist)
    
    # Vectorized, memory intensive
    comparison_matrix = np.repeat(train_dist.reshape(-1, 1), len(kth_neighbor_dist), axis=1)
    scores = np.mean(kth_neighbor_dist < comparison_matrix, axis=0)
    
    scores_full[:, search_set] = scores

scores = np.mean(scores_full, axis=1)

# %%
from sklearn import metrics

threshold = np.quantile(scores, portion_positive)
y_pred = np.where(scores > threshold, 1, 0)
y_true = np.concatenate([np.ones(N), np.zeros(N)])

metrics.accuracy_score(y_true, y_pred)

# %%
