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

# %%
from sklearn.decomposition import PCA

def PCA_by_variance(X_train, X_test, variance_threshold=0.9):
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train)

    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    are_features_enough = explained_variance >= variance_threshold
    num_features = np.where(are_features_enough)[0][0] + 1 if np.any(are_features_enough) else X_train.shape[1]
    X_train_pca = X_train_pca[:, :num_features]

    X_test_pca = pca.transform(X_test)
    X_test_pca = X_test_pca[:, :num_features]
    return X_train_pca, X_test_pca, explained_variance[:num_features]

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

# %%
from pyod.models.ecod import ECOD
from ecod_v2 import ECODv2
from ecod_v2_min import ECODv2Min
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

def get_occ_from_name(clf_name):
    if clf_name == 'ECOD':
        clf = PyODWrapper(ECOD())
    elif clf_name == 'ECODv2':
        clf = PyODWrapper(ECODv2())
    elif clf_name == 'ECODv2Min':
        clf = PyODWrapper(ECODv2Min())
    elif clf_name == 'GeomMedian':
        clf = GeomMedianDistance()
    elif clf_name == 'Mahalanobis':
        clf = Mahalanobis()
    elif clf_name == 'OC-SVM':
        clf = OneClassSVM()
    elif clf_name == 'IForest':
        clf = IsolationForest()
    return clf

# %%
def filter_inliers(X_test, y_test):
    inliers = np.where(y_test == 1)[0]
    y_test = y_test[inliers]
    X_test = X_test[inliers, :]
    return X_test,y_test

# %%
def apply_PCA_threshold(X_train_orig, X_test_orig, y_test_orig, pca_variance_threshold):
    X_train, X_test, y_test = X_train_orig, X_test_orig, y_test_orig
    if pca_variance_threshold is not None:
        X_train, X_test, _ = PCA_by_variance(X_train, X_test, pca_variance_threshold)
    
    return X_train, X_test, y_test

def apply_PCA_to_baseline(baseline):
    return baseline in ['ECODv2', 'Mahalanobis']

# %%
def prepare_resampling_threshold(clf, X_train, resampling_repeats, inlier_rate, method):
    N = len(X_train)
    thresholds = []

    for _ in range(resampling_repeats):
        if method == 'Bootstrap':
            resampling_samples = np.random.choice(range(N), size=N, replace=True)
        elif method == 'Multisplit':
            resampling_samples = np.random.choice(range(N), size=int(N/2), replace=False)
        else:
            raise NotImplementedError()
        
        is_selected_sample = np.isin(range(N), resampling_samples)
        X_resampling_train, X_resampling_cal = X_train[is_selected_sample], X_train[~is_selected_sample]
                            
        clf.fit(X_resampling_train)
        scores = clf.score_samples(X_resampling_cal)

        emp_quantile = np.quantile(scores, q=1 - inlier_rate)
        thresholds.append(emp_quantile)
    
    resampling_threshold = np.mean(thresholds)
    return resampling_threshold

# %%
def prepare_multisplit_cal_scores(clf, X_train, resampling_repeats):
    N = len(X_train)
    cal_scores_all = np.zeros((resampling_repeats, N - int(N/2)))

    for i in range(resampling_repeats):
        multisplit_samples = np.random.choice(range(N), size=int(N/2), replace=False)
        is_multisplit_sample = np.isin(range(N), multisplit_samples)
        X_multi_train, X_multi_cal = X_train[is_multisplit_sample], X_train[~is_multisplit_sample]
        
        clf.fit(X_multi_train)
        cal_scores = clf.score_samples(X_multi_cal)
        cal_scores_all[i, :] = cal_scores
    
    return cal_scores_all

def get_multisplit_p_values(scores, multisplit_cal_scores, median_multiplier=2):
    resampling_repeats = len(multisplit_cal_scores)

    p_vals_all = np.zeros((resampling_repeats, len(scores)))
    for i in range(resampling_repeats):
        cal_scores = multisplit_cal_scores[i, :]
        num_smaller_cal_scores = (scores > cal_scores.reshape(-1, 1)).sum(axis=0)
        p_vals = (num_smaller_cal_scores + 1) / (len(cal_scores) + 1)
        p_vals_all[i, :] = p_vals

    p_vals = median_multiplier * np.median(p_vals_all, axis=0)
    return p_vals

def apply_multisplit_to_baseline(baseline):
    return baseline in ['ECODv2', 'Mahalanobis']

# %%
def use_BH_procedure(p_vals, alpha, pi=None):
    if pi is None:
        fdr_ctl_threshold = alpha
    else:
        fdr_ctl_threshold = alpha / pi
    
    # if 'pi' in cutoff_type:
    #     pi = inlier_rate
    #     fdr_ctl_threshold = alpha / pi
                                
    sorted_indices = np.argsort(p_vals)
    bh_thresholds = np.linspace(fdr_ctl_threshold / len(p_vals), fdr_ctl_threshold, len(p_vals))
                                
    # is_h0_rejected == is_outlier, H_0: X ~ P_X
    is_h0_rejected = p_vals[sorted_indices] < bh_thresholds

    # take all the point to the left of last discovery
    rejections = np.where(is_h0_rejected)[0]
    if len(rejections) > 0:
        is_h0_rejected[:(np.max(rejections) + 1)] = True

    y_pred = np.ones_like(p_vals)
    y_pred[sorted_indices[is_h0_rejected]] = 0
    return y_pred

# %%
from sklearn import metrics

def get_metrics(y_test, y_pred, scores, one_class_only=False):
    false_detections = np.sum((y_pred == 0) & (y_test == 1))
    detections = np.sum(y_pred == 0)
    if detections == 0:
        fdr = np.nan
    else:
        fdr = false_detections / detections

    if not one_class_only:
        auc = metrics.roc_auc_score(y_test, scores)
    else:
        auc = np.nan
    
    acc = metrics.accuracy_score(y_test, y_pred)
    pre = metrics.precision_score(y_test, y_pred, zero_division=0)
    rec = metrics.recall_score(y_test, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_test, y_pred, zero_division=0)

    inlier_idx = np.where(y_test == 1)[0]
    t1e = 1 - np.mean(y_pred[inlier_idx] == y_test[inlier_idx])

    # important for PU
    false_rejections = np.sum((y_pred == 1) & (y_test == 0)) # False rejections == negative samples predicted to be positive
    rejections = np.sum(y_pred == 1) # All rejections == samples predicted to be positive
    if rejections == 0:
        frr = np.nan
    else:
        frr = false_rejections / rejections

    return {
        'AUC': auc,
        'ACC': acc,
        'PRE': pre,
        'REC': rec,
        'F1': f1,
        'FDR': fdr,
        'FRR': frr,
        'T1E': t1e, # Type I Error

        '#FD': false_detections,
        '#D': detections,
    }

# %%
import pandas as pd

def append_mean_row(df):
    name = ('Mean',) +  ('',) * (df.index.nlevels - 1) if df.index.nlevels > 1 else 'Mean'
    return pd.concat([
        df,
        df.mean(axis=0).to_frame(name=name).transpose().round(3)
    ])