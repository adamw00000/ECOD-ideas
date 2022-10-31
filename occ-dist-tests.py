# %%
import numpy as np
import matplotlib.pyplot as plt

#   autoregresja rzędu 1: sigma[i,j] => rho ^ |i-j|, rho \in (-1; 1)
#   Normalny(0, I) * sqrt(sigma)

n=1_000
dim=2
rho=0.5

def sample_autoregressive(n, dim=2, rho=0.5, diff=2):
    rho_matrix = rho * np.ones((dim, dim))
    rho_matrix

    i_mat = np.repeat(np.arange(dim).reshape(-1, 1), dim, axis=1)
    j_mat = np.repeat(np.arange(dim).reshape(1, -1), dim, axis=0)
    i_mat, j_mat

    autoregressive_matrix = rho_matrix ** np.abs(i_mat - j_mat)

    normal_sample_train = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), n)

    diff_mean = diff * np.ones(dim) * ((-1) ** np.array(range(dim)))
    normal_sample_test_inlier = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), n)
    normal_sample_test_outlier = np.random.multivariate_normal(diff_mean, 0.5 * np.eye(dim), n)
    normal_sample_test = np.concatenate([normal_sample_test_inlier, normal_sample_test_outlier])
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    return normal_sample_train @ autoregressive_matrix, \
        normal_sample_test @ autoregressive_matrix, \
        labels
        

auto_reg_train, auto_reg_test, _ = sample_autoregressive(n, dim, rho=0.5)
plt.plot(auto_reg_train[:, 0], auto_reg_train[:, 1], '.')
plt.plot(auto_reg_test[:, 0], auto_reg_test[:, 1], '.')

# %%
#   wykładniczy wielowymiarowy

def sample_exponential(n, dim=2, diff=2):
    train = np.concatenate([
        np.random.exponential(size=n).reshape(-1, 1)
        for _ in range(dim)
    ], axis = 1)

    test_inlier = np.concatenate([
        np.random.exponential(size=n).reshape(-1, 1)
        for _ in range(dim)
    ], axis = 1)
    test_outlier = np.concatenate([
        2 * np.random.exponential(size=n).reshape(-1, 1) + diff
        for _ in range(dim)
    ], axis = 1)

    test = np.concatenate([test_inlier, test_outlier])
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    return train, test, labels
    
multinomial_train, multinomial_test, _ = sample_exponential(n, dim)
plt.plot(multinomial_train[:, 0], multinomial_train[:, 1], '.')
plt.plot(multinomial_test[:, 0], multinomial_test[:, 1], '.')

# %%
def sample_normal(n, dim=2, diff=2):
    train = np.concatenate([
        np.random.randn(n).reshape(-1, 1)
        for _ in range(dim)
    ], axis=1)

    test_inlier = np.concatenate([
        np.random.randn(n).reshape(-1, 1)
        for _ in range(dim)
    ], axis=1)
    test_outlier = np.concatenate([
        0.5 * np.random.randn(n).reshape(-1, 1) + diff
        for _ in range(dim)
    ], axis=1)

    test = np.concatenate([test_inlier, test_outlier])
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    return train, test, labels
    
normal_train, normal_test, _ = sample_normal(n, dim)
plt.plot(normal_train[:, 0], normal_train[:, 1], '.')
plt.plot(normal_test[:, 0], normal_test[:, 1], '.')

# %%
from sklearn.decomposition import PCA

def PCA_by_variance(X_train, X_test, variance_threshold=0.9):
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train)

    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    are_features_enough = explained_variance >= variance_threshold
    num_features = np.where(are_features_enough)[0][0] + 1 if np.any(are_features_enough) else X.shape[1]
    X_train_pca = X_train_pca[:, :num_features]

    X_test_pca = pca.transform(X_test)
    X_test_pca = X_test_pca[:, :num_features]
    return X_train_pca, X_test_pca, explained_variance[:num_features]

X_train_pca, X_test_pca, variance = PCA_by_variance(auto_reg_train, auto_reg_test)
variance

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

class GeomMedianDistance():
    def fit(self, X, eps=1e-5):
        self.median = geometric_median(X, eps)
        return self
    
    def score_samples(self, X):
        return -np.linalg.norm(X - self.median, 2, axis=1)

clf = GeomMedianDistance()
clf.fit(auto_reg_train)

plt.scatter(auto_reg_train[:, 0], auto_reg_train[:, 1], c=clf.score_samples(auto_reg_train))
plt.show()
plt.scatter(auto_reg_test[:, 0], auto_reg_test[:, 1], c=clf.score_samples(auto_reg_test))
plt.show()

# %%
import numpy as np

def dot_diag(A, B):
    # Diagonal of the matrix product
    # equivalent to: np.diag(A @ B)
    return np.einsum('ij,ji->i', A, B)

class Mahalanobis():
    def fit(self, X):
        self.mu = np.mean(X, axis=0).reshape(1, -1)

        if X.shape[1] != 1:
            # sometimes non invertible
            # self.sigma_inv = np.linalg.inv(np.cov(X.T))

            # use pseudoinverse
            self.sigma_inv = np.linalg.pinv(np.cov(X.T))
            # another idea: add small number to diagonal
            # self.sigma_inv = np.linalg.inv(np.cov(X.T) + EPS * np.eye(X.shape[1]))
        else:
            self.sigma_inv = np.eye(1)
            
        return self
    
    def score_samples(self, X):
        # (X - self.mu) @ self.sigma_inv @ (X - self.mu).T
        # but we need only the diagonal
        mahal = dot_diag((X - self.mu) @ self.sigma_inv, (X - self.mu).T)
        return 1 / (1 + mahal)

clf = Mahalanobis()
clf.fit(auto_reg_train)
scores = clf.score_samples(auto_reg_train)

plt.scatter(auto_reg_train[:, 0], auto_reg_train[:, 1], c=clf.score_samples(auto_reg_train))
plt.show()
plt.scatter(auto_reg_test[:, 0], auto_reg_test[:, 1], c=clf.score_samples(auto_reg_test))
plt.show()

# %%
from pyod.models.ecod import ECOD
from ecod_v2 import ECODv2

class PyODWrapper():
    def __init__(self, model):
        self.model = model
    
    def fit(self, X_train):
        self.model.fit(X_train)
        return self

    def score_samples(self, X):
        return -self.model.decision_function(X)

clf = PyODWrapper(ECODv2())
clf.fit(auto_reg_train)
scores = clf.score_samples(auto_reg_train)

plt.scatter(auto_reg_train[:, 0], auto_reg_train[:, 1], c=clf.score_samples(auto_reg_train))
plt.show()
plt.scatter(auto_reg_test[:, 0], auto_reg_test[:, 1], c=clf.score_samples(auto_reg_test))
plt.show()

# %%
import os
import occ_datasets
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

from pyod.models.ecod import ECOD
from ecod_v2 import ECODv2
from ecod_v2_min import ECODv2Min
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

n_repeats = 10
resampling_repeats = 10
RESULTS_DIR = 'resultsdist'
os.makedirs(RESULTS_DIR, exist_ok=True)

full_results = []

for distribution, get_data in [
    ('Normal', sample_normal),
    ('Exponential', sample_exponential),
    ('Autoregressive', sample_autoregressive)
]:
    results = []

    for num_samples in [100, 10_000]:
        for dim in [2, 10, 50]:
            for exp in range(n_repeats):
                # Load data
                X_train, X_test, y_test = get_data(num_samples, dim)
                inlier_rate = np.mean(y_test)

                for baseline in [
                    'ECOD',
                    'ECODv2',
                    'ECODv2Min',
                    'GeomMedian',
                    'Mahalanobis',
                    'OC-SVM',
                    'IForest',
                ]:
                    for pca_variance_threshold in [0.5, 0.9, None]:
                        if pca_variance_threshold is not None:
                            if not 'ECODv2' in baseline:
                                continue
                            X_train, X_test, _ = PCA_by_variance(X_train, X_test, pca_variance_threshold)

                        if baseline == 'ECOD':
                            clf = PyODWrapper(ECOD())
                        elif baseline == 'ECODv2':
                            clf = PyODWrapper(ECODv2())
                        elif baseline == 'ECODv2Min':
                            clf = PyODWrapper(ECODv2Min())
                        elif baseline == 'GeomMedian':
                            clf = GeomMedianDistance()
                        elif baseline == 'Mahalanobis':
                            clf = Mahalanobis()
                        elif baseline == 'OC-SVM':
                            clf = OneClassSVM()
                        elif baseline == 'IForest':
                            clf = IsolationForest()
                        
                        for cutoff_type in [
                            'Empirical',
                            'Chi-squared',
                            'Bootstrap',
                            'Multisplit'
                        ]:
                            if cutoff_type != 'Empirical' and not 'ECODv2' in baseline:
                                continue
                            
                            N = len(X_train)
                            if cutoff_type == 'Bootstrap' or cutoff_type == 'Multisplit':
                                thresholds = []

                                for i in range(resampling_repeats):
                                    if cutoff_type == 'Bootstrap':
                                        resampling_samples = np.random.choice(range(N), size=N, replace=True)
                                    else:
                                        # Multisplit
                                        resampling_samples = np.random.choice(range(N), size=int(N/2), replace=False)
                                    is_selected_sample = np.isin(range(N), resampling_samples)
                                    X_resampling_train, X_resampling_cal = X_train[is_selected_sample], X_train[~is_selected_sample]
                                    
                                    clf.fit(X_resampling_train)
                                    scores = clf.score_samples(X_resampling_cal)

                                    emp_quantile = np.quantile(scores, q=1 - inlier_rate)
                                    thresholds.append(emp_quantile)                        
                                resampling_threshold = np.mean(thresholds)

                            clf.fit(X_train)

                            scores = clf.score_samples(X_test)
                            auc = metrics.roc_auc_score(y_test, scores)

                            if cutoff_type == 'Empirical':
                                emp_quantile = np.quantile(scores, q=1 - inlier_rate)
                                y_pred = np.where(scores > emp_quantile, 1, 0)
                            elif cutoff_type == 'Chi-squared':
                                d = X_test.shape[1]
                                chi_quantile = -scipy.stats.chi2.ppf(1 - inlier_rate, 2 * d)
                                y_pred = np.where(scores > chi_quantile, 1, 0)
                            elif cutoff_type == 'Bootstrap' or cutoff_type == 'Multisplit':
                                y_pred = np.where(scores > resampling_threshold, 1, 0)
                        
                            false_detections = np.sum((y_pred == 0) & (y_test == 1))
                            detections = np.sum(y_pred == 0)
                            fdr = false_detections / detections

                            acc = metrics.accuracy_score(y_test, y_pred)
                            pre = metrics.precision_score(y_test, y_pred)
                            rec = metrics.recall_score(y_test, y_pred)
                            f1 = metrics.f1_score(y_test, y_pred)

                            print(f'{distribution} ({num_samples}x{dim}): {baseline}{f"+PCA{pca_variance_threshold:.1f}" if pca_variance_threshold is not None else ""} ({cutoff_type}, {exp+1}/{n_repeats})' + \
                                f' ||| AUC: {100 * auc:3.2f}, ACC: {100 * acc:3.2f}, F1: {100 * f1:3.2f}')
                            occ_metrics = {
                                'Distribution': distribution,
                                'N': num_samples,
                                'Dim': dim,
                                'Method': baseline + (f"+PCA{pca_variance_threshold:.1f}" if pca_variance_threshold is not None else ""),
                                'Cutoff': cutoff_type,
                                'Exp': exp + 1,
                                'AUC': auc,
                                'ACC': acc,
                                'PRE': pre,
                                'REC': rec,
                                'F1': f1,
                                'FDR': fdr,
                            }
                            results.append(occ_metrics)
                            full_results.append(occ_metrics)
    
    df = pd.DataFrame.from_records(results)

    dist_df = df[df.Distribution == distribution]
    res_df = dist_df.groupby(['Distribution', 'N', 'Dim', 'Method', 'Cutoff'])\
        [['AUC', 'ACC', 'PRE', 'REC', 'F1', 'FDR']] \
        .mean() \
        .round(4) \
        * 100
    display(res_df)
    res_df.to_csv(os.path.join(RESULTS_DIR, f'dist-{distribution}.csv'))

# Full result pivots
df = pd.DataFrame.from_records(full_results)
df

for metric in ['AUC', 'ACC', 'F1', 'PRE', 'REC', 'FDR']:
    metric_df = df
    if metric == 'AUC':
        metric_df = df.loc[df.Cutoff == 'Empirical']
    
    (metric_df \
        .pivot_table(values=metric, index=['Distribution', 'N', 'Dim'], columns=['Method', 'Cutoff']) \
        * 100) \
        .round(2) \
        .to_csv(os.path.join(RESULTS_DIR, f'dist-all-{metric}.csv'))

# %%
