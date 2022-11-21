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
            try:
                self.sigma_inv = np.linalg.pinv(np.cov(X.T))
            except:
                # another idea: add small number to diagonal
                EPS = 1e-5
                self.sigma_inv = np.linalg.inv(np.cov(X.T) + EPS * np.eye(X.shape[1]))
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
    # return baseline in ['ECODv2', 'Mahalanobis']
    return baseline in ['ECODv2']

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
def apply_multisplit_to_baseline(baseline):
    return baseline in ['ECODv2', 'Mahalanobis']

# %%
from sklearn import metrics

def get_metrics(y_true, y_pred, scores, pos_class_only=False, \
        default_pre=0, default_rec=0, default_f1=0, \
        default_fdr=0, default_fnr=0, default_for=0):
    false_detections = np.sum((y_pred == 0) & (y_true == 1))
    detections = np.sum(y_pred == 0)
    if detections != 0:
        fdr = false_detections / detections
    else:
        # fdr = np.nan
        fdr = default_fdr

    if not pos_class_only:
        auc = metrics.roc_auc_score(y_true, scores)
    else:
        auc = np.nan
    
    acc = metrics.accuracy_score(y_true, y_pred)
    pre = metrics.precision_score(y_true, y_pred, zero_division=default_pre)
    rec = metrics.recall_score(y_true, y_pred, zero_division=default_rec)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=default_f1)

    inlier_idx = np.where(y_true == 1)[0]
    t1e = 1 - np.mean(y_pred[inlier_idx] == y_true[inlier_idx])

    # important for PU
    false_rejections = np.sum((y_pred == 1) & (y_true == 0)) # False rejections == negative samples predicted to be positive
    rejections = np.sum(y_pred == 1) # All rejections == samples predicted to be positive
    if rejections != 0:
        false_omission_rate = false_rejections / rejections
    else:
        # false_omission_rate = np.nan
        false_omission_rate = default_for

    if np.sum(y_true == 0) != 0:
        fnr = false_rejections / np.sum(y_true == 0)
    else:
        # fnr = np.nan
        fnr = default_fnr

    return {
        'AUC': auc,
        'ACC': acc,
        'PRE': pre,
        'REC': rec,
        'F1': f1,
        'FDR': fdr,
        'FOR': false_omission_rate,
        'FNR': fnr,
        'T1E': t1e, # Type I Error

        '#FD': false_detections,
        '#D': detections,
    }

def prepare_metrics(y_test, y_pred, scores, occ_metrics, metric_list, pos_class_only=False):
    method_metrics = dict(occ_metrics)
    test_metrics = get_metrics(y_test, y_pred, scores, pos_class_only=pos_class_only)

    for metric in metric_list:
        if metric in method_metrics and metric not in test_metrics:
            continue
        method_metrics[metric] = test_metrics[metric]
    return method_metrics

# %%
from occ_cutoffs import *

def get_cutoff_predictions(cutoff, X_train, X_test, inlier_rate, visualize_tests=False, apply_control_cutoffs=False,
        control_cutoff_params=None, common_visualization_params=None, special_visualization_params=None):
    scores, y_pred = cutoff.fit_apply(X_train, X_test, inlier_rate)
    yield cutoff.cutoff_type, scores, y_pred
    
    if not isinstance(cutoff, MultisplitCutoff):
        return
    
    alpha, inlier_rate = \
        control_cutoff_params['alpha'], control_cutoff_params['inlier_rate']
    exp, pca_variance_threshold, X_train, X_test, y_test = \
        special_visualization_params['exp'], \
        special_visualization_params['pca_variance_threshold'], \
        special_visualization_params['X_train'], \
        special_visualization_params['X_test'], \
        special_visualization_params['y_test']

    # Multisplit only
    visualize = (visualize_tests and exp == 0 and pca_variance_threshold is None)
    if visualize:
        visualize_multisplit(cutoff, (X_train, X_test, y_test), \
            common_visualization_params)
        
        # Set up plots for later
        plot_infos = prepare_cutoff_plots(cutoff, **common_visualization_params)

    if not apply_control_cutoffs:
        return

    for cutoff_num, control_cutoff in enumerate([
        BenjaminiHochbergCutoff(cutoff, alpha, None),
        BenjaminiHochbergCutoff(cutoff, alpha, inlier_rate),
        FORControlCutoff(cutoff, alpha, inlier_rate),
        FNRControlCutoff(cutoff, alpha, inlier_rate),
        CombinedFORFNRControlCutoff(cutoff, alpha, inlier_rate),
    ]):
        scores, y_pred = control_cutoff.fit_apply(X_test)
        yield control_cutoff.full_cutoff_type, scores, y_pred

        if visualize:
            draw_cutoff_plots(control_cutoff, X_test, y_test, \
                common_visualization_params, plot_infos[cutoff_num])

def visualize_multisplit(cutoff, visualization_data, 
        common_visualization_params):
    cutoff.visualize_calibration(visualization_data, 
            **common_visualization_params)
    cutoff.visualize_lottery(visualization_data, 
            **common_visualization_params,
            max_samples=100)
    cutoff.visualize_roc(visualization_data,
            **common_visualization_params)

    cutoff.visualize_p_values(visualization_data,
            **common_visualization_params)

def prepare_cutoff_plots(cutoff, test_case_name, clf_name, RESULTS_DIR):
    sns.set_theme()
    title = f'{test_case_name} - {clf_name}, {cutoff.cutoff_type}'
        
    bh_fig, bh_axs = plt.subplots(2, 2, figsize=(24, 16))
    bh_fig.suptitle(title)
        
    for_fig, for_axs = plt.subplots(1, 2, figsize=(24, 8))
    for_fig.suptitle(title)

    fnr_fig, fnr_axs = plt.subplots(1, 2, figsize=(24, 8))
    fnr_fig.suptitle(title)

    for_fnr_fig, for_fnr_axs = plt.subplots(1, 2, figsize=(24, 8))
    for_fnr_fig.suptitle(title)

    plot_info = [ 
            # ((fig, axs), save_plot)
            ((bh_fig, bh_axs[0, :]), False), 
            ((bh_fig, bh_axs[1, :]), True),
            ((for_fig, for_axs), True),
            ((fnr_fig, fnr_axs), True),
            ((for_fnr_fig, for_fnr_axs), True),
        ]
    
    return plot_info

def draw_cutoff_plots(control_cutoff, X_test, y_test, common_visualization_params, plot_info):
    ((fig, axs), save_plot) = plot_info
    zoom_left = isinstance(control_cutoff, BenjaminiHochbergCutoff)

    figure = (fig, axs[0])
    zoom = False
    control_cutoff.visualize(X_test, y_test, figure, \
        **common_visualization_params, \
        zoom=zoom, zoom_left=zoom_left, save_plot=False
    )

    figure = (fig, axs[1])
    zoom = True
    save_plot = save_plot
    control_cutoff.visualize(X_test, y_test, figure, \
        **common_visualization_params, \
        zoom=zoom, zoom_left=zoom_left, save_plot=save_plot
    )

# %%
import pandas as pd

def append_mean_row(df):
    name = ('Mean',) +  ('',) * (df.index.nlevels - 1) if df.index.nlevels > 1 else 'Mean'
    return pd.concat([
        df,
        df.mean(axis=0).to_frame(name=name).transpose().round(3)
    ])

# %%
metrics_to_multiply_by_100 = ['AUC', 'ACC', 'PRE', 'REC', 'F1']

def round_and_multiply_metric(df, metric):
    if metric in metrics_to_multiply_by_100:
        df = (df * 100).round(2)
    else:
        df = df.round(3)
    return df
