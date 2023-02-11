# %%
import scipy.stats
import pandas as pd
from occ_cutoffs import *
import time
from sklearn import metrics
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from A3_adapter import A3Adapter
from pyod.models.cblof import CBLOF
from hbos import HBOS
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from ecod_v2_min import ECODv2Min
from ecod_v2 import ECODv2
from pyod.models.ecod import ECOD
from scipy.spatial.distance import cdist, euclidean
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

#   autoregresja rzędu 1: sigma[i,j] => rho ^ |i-j|, rho \in (-1; 1)
#   Normalny(0, I) * sqrt(sigma)

n = 1_000
dim = 2
rho = 0.5


def sample_autoregressive(n, dim=2, rho=0.5, diff=2):
    rho_matrix = rho * np.ones((dim, dim))
    rho_matrix

    i_mat = np.repeat(np.arange(dim).reshape(-1, 1), dim, axis=1)
    j_mat = np.repeat(np.arange(dim).reshape(1, -1), dim, axis=0)
    i_mat, j_mat

    autoregressive_matrix = rho_matrix ** np.abs(i_mat - j_mat)

    normal_sample_train = np.random.multivariate_normal(
        np.zeros(dim), np.eye(dim), n)

    diff_mean = diff * np.ones(dim) * ((-1) ** np.array(range(dim)))
    normal_sample_test_inlier = np.random.multivariate_normal(
        np.zeros(dim), np.eye(dim), n)
    normal_sample_test_outlier = np.random.multivariate_normal(
        diff_mean, 0.5 * np.eye(dim), n)
    normal_sample_test = np.concatenate(
        [normal_sample_test_inlier, normal_sample_test_outlier])
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
    ], axis=1)

    test_inlier = np.concatenate([
        np.random.exponential(size=n).reshape(-1, 1)
        for _ in range(dim)
    ], axis=1)
    test_outlier = np.concatenate([
        2 * np.random.exponential(size=n).reshape(-1, 1) + diff
        for _ in range(dim)
    ], axis=1)

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


def PCA_by_variance(X_train, X_test, variance_threshold=0.9):
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train)

    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    are_features_enough = explained_variance >= variance_threshold
    num_features = np.where(are_features_enough)[
        0][0] + 1 if np.any(are_features_enough) else X_train.shape[1]
    X_train_pca = X_train_pca[:, :num_features]

    X_test_pca = pca.transform(X_test)
    X_test_pca = X_test_pca[:, :num_features]
    return X_train_pca, X_test_pca, explained_variance[:num_features]


# %%


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
    def fit(self, X, y=None, eps=1e-5):
        self.median = geometric_median(X, eps)
        return self

    def score_samples(self, X):
        return -np.linalg.norm(X - self.median, 2, axis=1)


# %%


def dot_diag(A, B):
    # Diagonal of the matrix product
    # equivalent to: np.diag(A @ B)
    return np.einsum('ij,ji->i', A, B)


class Mahalanobis():
    def fit(self, X, y=None):
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
                self.sigma_inv = np.linalg.inv(
                    np.cov(X.T) + EPS * np.eye(X.shape[1]))
        else:
            self.sigma_inv = np.eye(1)

        return self

    def score_samples(self, X):
        # (X - self.mu) @ self.sigma_inv @ (X - self.mu).T
        # but we need only the diagonal
        mahal = dot_diag((X - self.mu) @ self.sigma_inv, (X - self.mu).T)
        return 1 / (1 + mahal)


# %%


class PyODWrapper():
    def __init__(self, model):
        self.model = model

    def fit(self, X_train, y_train=None):
        self.model.fit(X_train)
        return self

    def score_samples(self, X):
        return -self.model.decision_function(X)


# %%
# from pyod.models.hbos import HBOS


class CBLOFWrapper():
    def __init__(self, model):
        assert isinstance(model, CBLOF)
        self.model = model

    def fit(self, X_train, y_train=None):
        is_fitted = False
        while not is_fitted:
            try:
                self.model.fit(X_train)
                is_fitted = True
            except ValueError:
                self.model.n_clusters //= 2
                if self.model.n_clusters == 1:
                    self.model.n_clusters = 8
                    self.model.alpha *= 0.75
                    self.model.beta *= 0.75
        return self

    def score_samples(self, X):
        return -self.model.decision_function(X)


class OracleWrapper():
    def __init__(self, model):
        self.model = model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def score_samples(self, X):
        return self.model.predict_proba(X)[:, 1]


def get_occ_from_name(clf_name, random_state, RESULTS_DIR, contamination=0.001):
    if 'Oracle' in clf_name:
        if 'LR' in clf_name:
            return OracleWrapper(LogisticRegression())
        elif 'DT' in clf_name:
            return OracleWrapper(DecisionTreeClassifier())
        elif 'RF' in clf_name:
            return OracleWrapper(RandomForestClassifier())
        else:
            raise NotImplementedError('Oracle method not implemented')

    if clf_name == 'ECOD':
        return PyODWrapper(ECOD(contamination=contamination))
    elif clf_name == 'ECODv2':
        return PyODWrapper(ECODv2(contamination=contamination))
    elif clf_name == 'ECODv2Min':
        return PyODWrapper(ECODv2Min(contamination=contamination))
    elif clf_name == 'HBOS':
        return PyODWrapper(HBOS(n_bins='auto', contamination=contamination))
    elif clf_name == 'CBLOF':
        return CBLOFWrapper(CBLOF(contamination=contamination, random_state=random_state))
    elif clf_name == 'GeomMedian':
        return GeomMedianDistance()
    elif clf_name == 'Mahalanobis':
        return Mahalanobis()
    elif clf_name == 'OC-SVM':
        return OneClassSVM(nu=contamination)
    elif clf_name == 'IForest':
        return IsolationForest(contamination=contamination, random_state=random_state)
    elif clf_name == 'A^3':
        return A3Adapter(max_target_epochs=200, max_a3_epochs=1000, patience=30, verbose=0, model_dir=RESULTS_DIR)
    else:
        raise NotImplementedError('OCC method not implemented')


# %%


def serialize_clfs(cutoff, exp_dir, clf_name):
    clfs_path = os.path.join(exp_dir, 'clfs', cutoff.cutoff_type)
    os.makedirs(clfs_path, exist_ok=True)

    for i, clf in enumerate(cutoff.get_clfs()):
        if isinstance(clf, A3Adapter):
            model_dir = os.path.join(clfs_path, f'{clf_name}_{i}')
            clf.save(model_dir)
        else:
            joblib.dump(clf,
                        os.path.join(clfs_path, f'{clf_name}_{i}.joblib')
                        )

# %%


def save_experiment_data(exp_dir, X_train_orig, X_test_orig, y_test_orig, oracle_data):
    occ_data_path = os.path.join(exp_dir, 'occ-data')
    os.makedirs(occ_data_path, exist_ok=True)

    with open(os.path.join(occ_data_path, 'X_train.npy'), 'wb') as f:
        np.save(f, X_train_orig)
    with open(os.path.join(occ_data_path, 'X_test.npy'), 'wb') as f:
        np.save(f, X_test_orig)
    with open(os.path.join(occ_data_path, 'y_test.npy'), 'wb') as f:
        np.save(f, y_test_orig)

    oracle_data_path = os.path.join(exp_dir, 'oracle-data')
    os.makedirs(oracle_data_path, exist_ok=True)

    X_train, y_train, X_test, y_test = oracle_data

    with open(os.path.join(oracle_data_path, 'X_train.npy'), 'wb') as f:
        np.save(f, X_train)
    with open(os.path.join(oracle_data_path, 'y_train.npy'), 'wb') as f:
        np.save(f, y_train)
    with open(os.path.join(oracle_data_path, 'X_test.npy'), 'wb') as f:
        np.save(f, X_test)
    with open(os.path.join(oracle_data_path, 'y_test.npy'), 'wb') as f:
        np.save(f, y_test)

# %%


def filter_inliers(X_test, y_test):
    inliers = np.where(y_test == 1)[0]
    y_test = y_test[inliers]
    X_test = X_test[inliers, :]
    return X_test, y_test

# %%


def apply_PCA_threshold(X_train_orig, X_test_orig, y_test_orig, pca_variance_threshold):
    X_train, X_test, y_test = X_train_orig, X_test_orig, y_test_orig
    if pca_variance_threshold is not None:
        X_train, X_test, _ = PCA_by_variance(
            X_train, X_test, pca_variance_threshold)

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
            resampling_samples = np.random.choice(
                range(N), size=N, replace=True)
        elif method == 'Multisplit':
            resampling_samples = np.random.choice(
                range(N), size=int(N/2), replace=False)
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
    return baseline not in ['ECOD', 'ECODv2Min', 'GeomMedian']


# %%


def get_metrics(y_true, y_pred, scores, pos_class_only=False,
                default_pre=np.nan, default_rec=np.nan, default_f1=np.nan,
                default_fdr=np.nan, default_fnr=np.nan, default_for=np.nan,
                default_t1e=np.nan):
    AUC = metrics.roc_auc_score(
        y_true, scores) if not pos_class_only else np.nan

    # positives == "outliers" in this case
    PP = np.sum(y_pred == 0)  # predicted outliers (rejected samples)
    P = np.sum(y_true == 0)  # true outliers
    PN = np.sum(y_pred == 1)  # predicted inliers (not rejected samples)
    N = np.sum(y_true == 1)  # true inliers

    TP = np.sum((y_true == 0) & (y_pred == 0))  # correctly classified outliers
    # rejected inliers (incorrectly)
    FP = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 1) & (y_pred == 1))  # correctly classified inliers
    # not rejected outliers (incorrectly)
    FN = np.sum((y_true == 0) & (y_pred == 1))

    T1E = FP / N if N != 0 else default_t1e
    FDR = FP / PP if PP != 0 else default_fdr
    FOR = FN / PN if PN != 0 else default_for
    FNR = FN / P if P != 0 else default_fnr

    ACC = (TP + TN) / (P + N)

    PRE = TN / PN if PN != 0 else default_pre
    REC = TN / N if N != 0 else default_rec
    F1 = (2 * PRE * REC) / (PRE + REC) \
        if (PRE != 0 or REC != 0) and (not np.isnan(PRE) and not np.isnan(REC)) \
        else default_f1

    return {
        'AUC': AUC,
        'ACC': ACC,
        'PRE': PRE,
        'REC': REC,
        'F1':  F1,
        'FDR': FDR,  # False Discovery Rate
        'FOR': FOR,  # False Omission Rate
        'FNR': FNR,  # False Negative Rate
        'T1E': T1E,  # Type I Error

        '#TP': TP,
        '#FP': FP,
        '#TN': TN,
        '#FN': FN,

        'NoRejected': PN == 0,
        'NoAccepted': PP == 0,
    }


def prepare_metrics(y_test, y_pred, scores, occ_metrics, metric_list, pos_class_only=False):
    method_metrics = dict(occ_metrics)
    test_metrics = get_metrics(
        y_test, y_pred, scores, pos_class_only=pos_class_only)

    for metric in metric_list:
        if metric in method_metrics and metric not in test_metrics:
            continue
        method_metrics[metric] = test_metrics[metric]
    return method_metrics


# %%
def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array = array + 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))


def get_cutoff_predictions(cutoff, X_train, X_test, inlier_rate,
                           visualize_tests=False, apply_control_cutoffs=False, control_cutoff_params=None,
                           common_visualization_params=None, special_visualization_params=None,
                           y_train=None, y_test=None, exp_dir=None, clf_name=None):
    start_time = time.perf_counter()
    scores, y_pred = cutoff.fit_apply(
        X_train, X_test, inlier_rate, y_train=y_train)
    elapsed = time.perf_counter() - start_time
    yield cutoff.cutoff_type, scores, y_pred, elapsed

    # serialize_clfs(cutoff, exp_dir, clf_name)

    alpha, inlier_rate = \
        control_cutoff_params['alpha'], control_cutoff_params['inlier_rate']
    exp, pca_variance_threshold = \
        special_visualization_params['exp'], \
        special_visualization_params['pca_variance_threshold']

    if not (isinstance(cutoff, MultisplitCutoff)
            or isinstance(cutoff, NoSplitCutoff)):
        return

    # Multisplit only
    visualize = (visualize_tests and exp == 0 and pca_variance_threshold is None
                 and isinstance(cutoff, MultisplitCutoff) and cutoff.resampling_repeats != 1)
    if visualize:
        visualize_multisplit(cutoff, (X_train, y_train, X_test, y_test),
                             alpha, common_visualization_params)

        # Set up plots for later
        if apply_control_cutoffs:
            plot_infos = prepare_cutoff_plots(
                cutoff, **common_visualization_params)

    if not apply_control_cutoffs:
        return

    if isinstance(cutoff, MultisplitCutoff):
        p_vals = cutoff.get_p_vals(X_test)
        test_case_name, RESULTS_DIR = \
            common_visualization_params['test_case_name'], \
            common_visualization_params['RESULTS_DIR']
        inlier_idx = np.where(y_test == 1)[0]
        inlier_mask = np.isin(range(len(y_test)), inlier_idx)

        inliers_skew = scipy.stats.skew(p_vals[inlier_mask])
        outliers_skew = scipy.stats.skew(p_vals[~inlier_mask])

        inliers_kurt = scipy.stats.kurtosis(p_vals[inlier_mask])
        outliers_kurt = scipy.stats.kurtosis(p_vals[~inlier_mask])

        inliers_ks = scipy.stats.kstest(p_vals[inlier_mask], scipy.stats.uniform.cdf)
        outliers_ks = scipy.stats.kstest(p_vals[~inlier_mask], scipy.stats.uniform.cdf)

        os.makedirs(os.path.join(RESULTS_DIR, test_case_name), exist_ok=True)

        pval_metrics = [
            { 'Type': 'Inlier', 'Metric': 'Skewness', 'Value': inliers_skew },
            { 'Type': 'Outlier', 'Metric': 'Skewness', 'Value': outliers_skew },
            { 'Type': 'Inlier', 'Metric': 'Kurtosis', 'Value': inliers_kurt },
            { 'Type': 'Outlier', 'Metric': 'Kurtosis', 'Value': outliers_kurt },
            { 'Type': 'Inlier', 'Metric': 'KSStat', 'Value': inliers_ks.statistic },
            { 'Type': 'Outlier', 'Metric': 'KSStat', 'Value': outliers_ks.statistic },
            { 'Type': 'Inlier', 'Metric': 'KSpVal', 'Value': inliers_ks.pvalue },
            { 'Type': 'Outlier', 'Metric': 'KSpVal', 'Value': outliers_ks.pvalue },
        ]

        if os.path.exists(os.path.join(RESULTS_DIR, test_case_name, 'pval_metrics_v2.csv')):
            metrics_df = pd.read_csv(os.path.join(RESULTS_DIR, test_case_name, 'pval_metrics_v2.csv'))
        else:
            metrics_df = pd.DataFrame()

        pd.concat([
            metrics_df,
            pd.DataFrame.from_records(pval_metrics),
        ]).to_csv(
            os.path.join(RESULTS_DIR, test_case_name, 'pval_metrics_v2.csv'), index=False
        )

    for cutoff_num, control_cutoff in enumerate([
        # BenjaminiHochbergCutoff(cutoff, alpha, None),
        # BenjaminiHochbergCutoff(cutoff, alpha, inlier_rate),
        FORControlCutoffOld(cutoff, alpha, inlier_rate),
        # FORControlCutoffGreaterThan(cutoff, alpha, inlier_rate),
        # FORControlCutoffGreaterEqual(cutoff, alpha, inlier_rate),
        # FORControlCutoffGreaterEqualAcceptEqual(cutoff, alpha, inlier_rate),
        # FNRControlCutoff(cutoff, alpha, inlier_rate),
        # CombinedFORFNRControlCutoff(cutoff, alpha, inlier_rate),
    ]):
        start_time = time.perf_counter()
        scores, y_pred = control_cutoff.fit_apply(X_test)
        elapsed = time.perf_counter() - start_time
        yield control_cutoff.full_cutoff_type, scores, y_pred, elapsed

        if visualize:
            draw_cutoff_plots(control_cutoff, X_test, y_test,
                              common_visualization_params, plot_infos[cutoff_num])


def visualize_multisplit(cutoff, visualization_data,
                         alpha, common_visualization_params):
    if alpha != 0.25:
        cutoff.visualize_lottery(visualization_data,
                                 **common_visualization_params,
                                 alpha=alpha,
                                 max_samples=100)
        cutoff.visualize_calibration(visualization_data,
                                     **common_visualization_params)
        cutoff.visualize_roc(visualization_data,
                             **common_visualization_params)
        cutoff.visualize_p_values(visualization_data,
                                  **common_visualization_params)


def prepare_cutoff_plots(cutoff, test_case_name, clf_name, RESULTS_DIR):
    sns.set_theme()
    title = f'{test_case_name} - {clf_name}, {cutoff.cutoff_type}'

    # bh_fig, bh_axs = plt.subplots(2, 2, figsize=(24, 16))
    # bh_fig.suptitle(title)

    for_old_fig, for_old_axs = plt.subplots(1, 2, figsize=(24, 8))
    for_old_fig.suptitle(title)

    # for_gt_fig, for_gt_axs = plt.subplots(1, 2, figsize=(24, 8))
    # for_gt_fig.suptitle(title)

    # for_geq_fig, for_geq_axs = plt.subplots(1, 2, figsize=(24, 8))
    # for_geq_fig.suptitle(title)

    # for_geq_aeq_fig, for_geq_aeq_axs = plt.subplots(1, 2, figsize=(24, 8))
    # for_geq_aeq_fig.suptitle(title)

    # fnr_fig, fnr_axs = plt.subplots(1, 2, figsize=(24, 8))
    # fnr_fig.suptitle(title)

    # for_fnr_fig, for_fnr_axs = plt.subplots(1, 2, figsize=(24, 8))
    # for_fnr_fig.suptitle(title)

    plot_info = [
        # ((fig, axs), save_plot)
        # ((bh_fig, bh_axs[0, :]), False),
        # ((bh_fig, bh_axs[1, :]), True),
        ((for_old_fig, for_old_axs), True),
        # ((for_gt_fig, for_gt_axs), True),
        # ((for_geq_fig, for_geq_axs), True),
        # ((for_geq_aeq_fig, for_geq_aeq_axs), True),
        # ((fnr_fig, fnr_axs), True),
        # ((for_fnr_fig, for_fnr_axs), True),
    ]

    return plot_info


def draw_cutoff_plots(control_cutoff, X_test, y_test, common_visualization_params, plot_info):
    ((fig, axs), save_plot) = plot_info
    zoom_left = isinstance(control_cutoff, BenjaminiHochbergCutoff)

    plt.close(fig)

    # figure = (fig, axs[0])
    # zoom = False

    sns.set_theme()
    fig = plt.figure(figsize=(10, 8))
    figure = (fig, plt.gca())
    zoom = False

    control_cutoff.visualize(X_test, y_test, figure,
                             **common_visualization_params,
                             zoom=zoom, zoom_left=zoom_left,
                             # save_plot=False,
                             save_plot=True,
                             )

    # figure = (fig, axs[1])
    # zoom = True
    # save_plot = save_plot
    # control_cutoff.visualize(X_test, y_test, figure, \
    #     **common_visualization_params, \
    #     zoom=zoom, zoom_left=zoom_left, save_plot=save_plot
    # )


# %%


def append_mean_row(df):
    name = ('Mean',) + ('',) * (df.index.nlevels -
                                1) if df.index.nlevels > 1 else 'Mean'
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


# %%
default_metric_values = {
    'PRE': 0,
    'REC': 0,
    'F1': 0,
    'FDR': 0,
    'FOR': 0,
    'BFOR': 0,
    'FNR': 0,
    'T1E': 0,
}


def fill_nan_values(df, default_values=default_metric_values):
    for metric, default in default_values.items():
        if metric not in df.columns:
            continue

        df[metric] = df[metric].fillna(default)

    return df


# %%


def aggregate_results(df: pd.DataFrame, metric_list, alpha_metrics, RESULTS_DIR, DATASET_TYPE, alpha):
    mean_pivot_dict = {}
    for metric in metric_list:
        metric_df = df

        mean_pivot = metric_df \
            .pivot_table(
                values=metric, index=['Dataset'], columns=['Method', 'Cutoff'],
                dropna=False)
        sem_pivot = metric_df \
            .pivot_table(
                values=metric, index=['Dataset'], columns=['Method', 'Cutoff'],
                dropna=False, aggfunc=scipy.stats.sem)

        mean_pivot_dict[metric] = mean_pivot
        process_pivot(metric, mean_pivot, 'mean', alpha_metrics,
                      RESULTS_DIR, DATASET_TYPE, alpha)
        process_pivot(metric, sem_pivot, 'sem', alpha_metrics,
                      RESULTS_DIR, DATASET_TYPE, alpha)

    # BFOR is special - it operates on means instead of specific values
    if 'FOR' in metric_list and '#FN' in metric_list and '#TN' in metric_list:
        metric = 'BFOR'
        if metric not in metric_list:
            metric_list.append('metric')

        tn = mean_pivot_dict['#TN']
        fn = mean_pivot_dict['#FN']

        BFOR_pivot = fn / (tn + fn)
        BFOR_pivot = BFOR_pivot.fillna(default_metric_values[metric])
        mean_pivot_dict[metric] = BFOR_pivot

        if 'FOR' in alpha_metrics:
            alpha_metrics.append('BFOR')

        process_pivot(metric, BFOR_pivot, 'mean', alpha_metrics,
                      RESULTS_DIR, DATASET_TYPE, alpha)


def process_pivot(metric, mean_pivot, pivot_type, alpha_metrics, RESULTS_DIR, DATASET_TYPE, alpha):
    mean_pivot = mean_pivot.dropna(how='all')
    mean_pivot = append_mean_row(mean_pivot)
    mean_pivot = round_and_multiply_metric(mean_pivot, metric)

    mean_pivot \
        .to_csv(os.path.join(RESULTS_DIR, f'{DATASET_TYPE}-all-{metric}-{pivot_type}.csv'))
    mean_pivot \
        .transpose() \
        .to_csv(os.path.join(RESULTS_DIR, f'{DATASET_TYPE}-all-{metric}-{pivot_type}-transposed.csv'))

    if pivot_type == 'mean' and metric in alpha_metrics:
        append_mean_row(mean_pivot <= alpha) \
            .to_csv(os.path.join(RESULTS_DIR, f'{DATASET_TYPE}-all-{metric}-alpha.csv'))
        append_mean_row(mean_pivot <= alpha) \
            .transpose() \
            .to_csv(os.path.join(RESULTS_DIR, f'{DATASET_TYPE}-all-{metric}-alpha-transposed.csv'))
        # 2 * alpha threshold
        append_mean_row(mean_pivot <= 2 * alpha) \
            .to_csv(os.path.join(RESULTS_DIR, f'{DATASET_TYPE}-all-{metric}-2-alpha.csv'))
        append_mean_row(mean_pivot <= 2 * alpha) \
            .transpose() \
            .to_csv(os.path.join(RESULTS_DIR, f'{DATASET_TYPE}-all-{metric}-2-alpha-transposed.csv'))

# %%


def convert_occ_dataset_to_binary(X_train_occ, X_test_occ, y_test_occ, train_outlier_portion=0.6):
    outlier_idx = np.where(y_test_occ == 0)[0]

    num_training_outliers = int(train_outlier_portion * len(outlier_idx))
    train_outlier_idx = np.random.choice(
        outlier_idx, size=num_training_outliers, replace=False)
    train_outlier_mask = np.where(
        np.isin(range(len(y_test_occ)), train_outlier_idx), True, False)

    X_train = np.concatenate([
        X_train_occ,
        X_test_occ[train_outlier_mask],
    ])
    y_train = np.concatenate([
        np.ones(len(X_train_occ)),
        np.zeros(len(train_outlier_idx)),
    ])

    X_test = X_test_occ[~train_outlier_mask]
    y_test = y_test_occ[~train_outlier_mask]

    return X_train, y_train, X_test, y_test
