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
    # return baseline in ['ECODv2', 'Mahalanobis']
    return baseline in ['Mahalanobis']

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

def get_metrics(y_test, y_pred, scores, pos_class_only=False):
    false_detections = np.sum((y_pred == 0) & (y_test == 1))
    detections = np.sum(y_pred == 0)
    if detections == 0:
        fdr = np.nan
    else:
        fdr = false_detections / detections

    if not pos_class_only:
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
        false_omission_rate = np.nan
    else:
        false_omission_rate = false_rejections / rejections

    return {
        'AUC': auc,
        'ACC': acc,
        'PRE': pre,
        'REC': rec,
        'F1': f1,
        'FDR': fdr,
        'FOR': false_omission_rate,
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
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_scores(scores, p_vals, y_test, 
        train_scores, train_p_vals, test_case_name, 
        clf_name, cutoff_type, results_dir, plot_scores=True):
    inlier_idx = np.where(y_test == 1)[0]
    inlier_mask = np.isin(range(len(y_test)), inlier_idx)

    df = pd.DataFrame({
        'Score': scores,
        'p-value': p_vals,
        'Class': np.where(inlier_mask, 'Inlier', 'Outlier'),
    })
    train_df = pd.DataFrame({
        'Score': train_scores,
        'p-value': train_p_vals,
        'Class': np.array(['Inlier'] * len(train_scores))
    })

    os.makedirs(os.path.join(results_dir, 'img', test_case_name), exist_ok=True)

    for metric in ['Score', 'p-value']:
        if metric == 'Score' and not plot_scores:
            continue

        sns.set_theme()
        _, axs = plt.subplots(1, 2, figsize=(14, 6), 
            sharex=True, sharey=True)

        sns.histplot(train_df, x=metric, hue='Class', ax=axs[0],
            hue_order=['Inlier'], stat='probability')
        axs[0].set_title('Train')
        sns.histplot(df, x=metric, hue='Class', ax=axs[1],
            hue_order=['Inlier', 'Outlier'], stat='probability')
        axs[1].set_title('Test')
        
        plt.suptitle(f'{test_case_name} ({clf_name}, {cutoff_type}) - {metric} distribution')
        plt.savefig(
            os.path.join(results_dir, 'img', test_case_name, f'distribution-{metric}-{clf_name}-{cutoff_type}.png'),
            dpi=150,
            bbox_inches='tight',
            facecolor='white',
        )
        plt.close()

        if not plot_scores:
            continue

        # Plot ROC
        fpr, tpr, _ = metrics.roc_curve(y_test, df[metric], pos_label=1)

        sns.set_theme()
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label="ROC curve (area = %0.2f)" % metrics.auc(fpr, tpr),
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.01])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{test_case_name} ({clf_name}) - {metric} ROC (for Inliers)")
        plt.legend(loc="lower right")

        plt.savefig(
            os.path.join(results_dir, 'img', test_case_name, f'ROC-{metric}-{clf_name}-{cutoff_type}.png'),
            dpi=150,
            bbox_inches='tight',
            facecolor='white',
        )
        plt.close()

# %%
import matplotlib.pyplot as plt

def use_BH_procedure(p_vals, alpha, pi=None, 
        visualize=False, y_test=None, test_case_name=None,
        clf_name=None, cutoff_type=None, results_dir=None,
        bh_plot=None, save_plot=False):
    if pi is None:
        fdr_ctl_threshold = alpha
    else:
        fdr_ctl_threshold = alpha / pi
    
    sorted_indices = np.argsort(p_vals)
    bh_thresholds = np.linspace(fdr_ctl_threshold / len(p_vals), fdr_ctl_threshold, len(p_vals))
    
    # is_h0_rejected == is_outlier, H_0: X ~ P_X
    is_h0_rejected = p_vals[sorted_indices] < bh_thresholds

    # take all the point to the left of last discovery
    rejections = np.where(is_h0_rejected)[0]
    if len(rejections) > 0:
        num_rejected = (np.max(rejections) + 1)
        is_h0_rejected[:num_rejected] = True

    if visualize:
        visualize_BH(p_vals, alpha, pi, y_test, test_case_name, clf_name, cutoff_type,
            results_dir, fdr_ctl_threshold, sorted_indices,
            bh_thresholds, is_h0_rejected, rejections, 
            bh_plot, save_plot)

    y_pred = np.ones_like(p_vals)
    y_pred[sorted_indices[is_h0_rejected]] = 0
    return y_pred

def visualize_BH(p_vals, alpha, pi,
        y_test, test_case_name,
        clf_name, cutoff_type, results_dir, 
        fdr_ctl_threshold, sorted_indices, 
        bh_thresholds, is_h0_rejected, rejections,
        bh_plot, save_plot):
    os.makedirs(os.path.join(results_dir, 'img', test_case_name), exist_ok=True)

    fig, axs = bh_plot

    for zoom in [False, True]:
        num_elements = len(y_test)
        if zoom:
            num_zoom_elements = 100
            if len(rejections) > 0:
                num_rejected = (np.max(rejections) + 1)
                num_zoom_elements = max(num_zoom_elements, 2 * num_rejected)
            num_elements = min(num_zoom_elements, len(y_test))
        
        plot_row = 0 if pi is None else 1
        plot_col = 0 if not zoom else 1
        title = f'BH' + ('+pi' if pi is not None else '') + \
            f', alpha={fdr_ctl_threshold:.3f}' + \
            f'{f" (zoomed)" if zoom else ""}'

        ax = axs[plot_row, plot_col]
        
        x = list(range(len(p_vals)))
        pval_vec = p_vals[sorted_indices]
        outlier_vec = np.where(y_test[sorted_indices] == 0, 'Outlier', 'Inlier')
        rejected_vec = np.where(is_h0_rejected, 'Rejected', 'Not rejected')

        # plt.figure(figsize=(12, 8))
        sns.scatterplot(x=x[:num_elements], y=pval_vec[:num_elements], 
                hue=outlier_vec[:num_elements], style=rejected_vec[:num_elements],
                hue_order=['Inlier', 'Outlier'],
                style_order=['Not rejected', 'Rejected'],
                edgecolor='k', linewidth=.3,
                ax=ax)
        sns.scatterplot(x=x[:num_elements], y=bh_thresholds[:num_elements], 
                hue=['B-H threshold'] * num_elements,
                palette=['r'],
                edgecolor=None, s=2,
                ax=ax)
        # sns.lineplot(x=x[:num_elements], y=bh_thresholds[:num_elements],
        #         hue=['B-H threshold'] * num_elements,
        #         palette=['r'],
        #         linestyle='--',
        #         ax=ax)
        
        # ax.set_title(f'{test_case_name} - {clf_name}, {cutoff_type}, alpha={alpha}' + \
        #         f'{f" (+pi => alpha={fdr_ctl_threshold:.3f}" if pi is not None else ""}' + \
        #         f'{f" (zoomed)" if zoom else ""}')
        ax.set_title(title)
        ax.set_ylim(0, None)
        
        legend = ax.legend()
        legend.legendHandles[-1]._sizes = [8.]

    if save_plot:
        fig.tight_layout()
        fig.savefig(
                os.path.join(results_dir, 'img', test_case_name, 
                    f'BH-{clf_name}-{cutoff_type}.png'),
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
            )
        plt.close(fig)

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
