import numpy as np
import scipy.stats
from statsmodels.distributions.empirical_distribution import ECDF

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import metrics


def select_indices(idx, *args):
    return tuple([x[idx] for x in args])


class Cutoff():
    is_fitted = False
    
    @property
    def cutoff_type(self):
        raise NotImplementedError('Use derived class for cutoff')

    def fit(self, scores) -> float:
        raise NotImplementedError('Use derived class for cutoff')
    
    def apply(self, scores):
        raise NotImplementedError('Use derived class for cutoff')

    def fit_apply(self, scores):
        self.fit(scores)
        return self.apply(scores)


class EmpiricalCutoff(Cutoff):
    cutoff_type = 'Empirical'

    def __init__(self, inlier_rate):
        self.inlier_rate = inlier_rate

    def fit(self, scores):
        self.threshold_ = np.quantile(scores, q=1 - self.inlier_rate)
        self.is_fitted = True
        return self.threshold_
    
    def apply(self, scores):
        y_pred = np.where(scores > self.threshold_, 1, 0)
        return y_pred


class ChiSquaredCutoff(Cutoff):
    cutoff_type = 'Chi-squared'

    def __init__(self, inlier_rate, d):
        self.inlier_rate = inlier_rate
        # d - number of features
        self.d = d

    def fit(self, scores):
        self.threshold_ = -scipy.stats.chi2.ppf(1 - self.inlier_rate, 2 * self.d)
        self.is_fitted = True
        return self.threshold_
    
    def apply(self, scores):
        y_pred = np.where(scores > self.threshold_, 1, 0)
        return y_pred


class __ResamplingThresholdCutoffBase(Cutoff):
    def __init__(self, inlier_rate, resampling_repeats, X_train, clf):
        self.inlier_rate = inlier_rate
        # d - number of features
        self.resampling_repeats = resampling_repeats
        self.X_train = X_train
        self.clf = clf

    def _choose_samples(self, N):
        raise NotImplementedError('Use derived class for cutoff')

    def __prepare_threshold(self):
        N = len(self.X_train)
        thresholds = []

        for _ in range(self.resampling_repeats):
            resampling_samples = self._choose_samples(N)

            is_selected_sample = np.isin(range(N), resampling_samples)
            X_resampling_train, X_resampling_cal = self.X_train[is_selected_sample], self.X_train[~is_selected_sample]

            self.clf.fit(X_resampling_train)
            scores = self.clf.score_samples(X_resampling_cal)

            emp_quantile = np.quantile(scores, q=1 - self.inlier_rate)
            thresholds.append(emp_quantile)
        
        resampling_threshold = np.mean(thresholds)
        return resampling_threshold

    def fit(self, scores):
        self.threshold_ = self.__prepare_threshold()
        self.is_fitted = True
        return self.threshold_

    def apply(self, scores):
        y_pred = np.where(scores > self.threshold_, 1, 0)
        return y_pred


class BootstrapThresholdCutoff(__ResamplingThresholdCutoffBase):
    cutoff_type = 'Bootstrap_threshold'

    def __init__(self, inlier_rate, resampling_repeats, X_train, clf):
        super().__init__(inlier_rate, resampling_repeats, X_train, clf)

    def _choose_samples(self, N):
        return np.random.choice(range(N), size=N, replace=True)

class MultisplitThresholdCutoff(__ResamplingThresholdCutoffBase):
    cutoff_type = 'Multisplit_threshold'

    def __init__(self, inlier_rate, resampling_repeats, X_train, clf):
        super().__init__(inlier_rate, resampling_repeats, X_train, clf)

    def _choose_samples(self, N):
        return np.random.choice(range(N), size=int(N/2), replace=False)


class MultisplitCutoff(Cutoff):
    @property
    def cutoff_type(self):
        return 'Multisplit' \
            + (f'-{self.median_multiplier}_median' if self.median_multiplier != 1 else '') \
            + (f'-{self.resampling_repeats}_repeat' if self.resampling_repeats != 10 else '')

    def __init__(self, inlier_rate, resampling_repeats, X_train, clf, alpha, median_multiplier=2):
        self.inlier_rate = inlier_rate
        # d - number of features
        self.resampling_repeats = resampling_repeats
        self.X_train = X_train
        self.clf = clf

        self.alpha = alpha
        self.median_multiplier = median_multiplier
    
    def __prepare_multisplit_cal_scores(self):
        N = len(self.X_train)
        cal_scores_all = np.zeros((self.resampling_repeats, N - int(N/2)))

        for i in range(self.resampling_repeats):
            multisplit_samples = np.random.choice(range(N), size=int(N/2), replace=False)
            is_multisplit_sample = np.isin(range(N), multisplit_samples)
            X_multi_train, X_multi_cal = self.X_train[is_multisplit_sample], self.X_train[~is_multisplit_sample]
            
            self.clf.fit(X_multi_train)
            cal_scores = self.clf.score_samples(X_multi_cal)
            cal_scores_all[i, :] = cal_scores
        
        return cal_scores_all

    def __get_multisplit_p_values(self, scores):
        resampling_repeats = len(self.cal_scores_)

        p_vals_all = np.zeros((resampling_repeats, len(scores)))
        for i in range(resampling_repeats):
            cal_scores = self.cal_scores_[i, :]
            num_smaller_cal_scores = (scores > cal_scores.reshape(-1, 1)).sum(axis=0)
            p_vals = (num_smaller_cal_scores + 1) / (len(cal_scores) + 1)
            p_vals_all[i, :] = p_vals

        p_vals = self.median_multiplier * np.median(p_vals_all, axis=0)
        return p_vals
    
    def fit(self, scores):
        self.cal_scores_ = self.__prepare_multisplit_cal_scores()
        self.is_fitted = True
        return self.cal_scores_

    def get_p_vals(self, scores):
        return self.__get_multisplit_p_values(scores)
    
    def apply_to_p_vals(self, p_vals):
        y_pred = np.where(p_vals < self.alpha, 0, 1)
        return y_pred
    
    def apply(self, scores):
        p_vals = self.__get_multisplit_p_values(scores)
        return self.apply_to_p_vals(p_vals)
    
    def visualize_lottery(self, scores, y_test, 
            test_case_name, clf_name, RESULTS_DIR, 
            max_samples=100):
        resampling_repeats = len(self.cal_scores_)

        p_vals_all = np.zeros((resampling_repeats, len(scores)))
        for i in range(resampling_repeats):
            cal_scores = self.cal_scores_[i, :]
            num_smaller_cal_scores = (scores > cal_scores.reshape(-1, 1)).sum(axis=0)
            p_vals = (num_smaller_cal_scores + 1) / (len(cal_scores) + 1)
            p_vals_all[i, :] = p_vals

        p_vals = self.median_multiplier * np.median(p_vals_all, axis=0)

        p_val_range = p_vals_all.max(axis=0) - p_vals_all.min(axis=0)
        mean_range = np.mean(p_val_range)
        median_range = np.median(p_val_range)
        max_range = np.max(p_val_range)
        mean_start = np.mean(p_vals_all.min(axis=0))
        mean_end = np.mean(p_vals_all.max(axis=0))

        os.makedirs(os.path.join(RESULTS_DIR, 'img', test_case_name), exist_ok=True)

        plt.figure(figsize=(16, 6))
        sns.set_theme()

        data = p_vals_all[:, :max_samples]
        sample_types = np.where(y_test == 1, 'Inlier', 'Outlier')
        hue_order = ['Inlier', 'Outlier']

        sns.scatterplot(
            x=np.array(range(len(p_vals)))[:max_samples],
            y=p_vals[:max_samples],
            hue=sample_types[:max_samples],
            hue_order=hue_order,
            palette=[sns.color_palette()[0], sns.color_palette()[3]],
            zorder=100, s=12, edgecolor='k',
        )
        plt.legend(title='Median values')

        df = pd.DataFrame({
            'p-value': p_vals_all[:, :max_samples].reshape(-1),
            'Split': np.repeat(range(1, 11), data.shape[1]),
            'Sample': np.tile(range(1, data.shape[1] + 1), data.shape[0]),
        })        
        sns.stripplot(data=df, x='Sample', y='p-value', linewidth=0.2)

        plt.tick_params(axis='x', labelsize=6)
        plt.title(f'{self.cutoff_type} ({clf_name}) p-value lottery' + \
            f' - max range length: {max_range:.3f}, mean: {mean_range:.3f}, median: {median_range:.3f}, range: ({mean_start:.3f}-{mean_end:.3f})')

        plt.savefig(
            os.path.join(RESULTS_DIR, 'img', test_case_name, f'lottery-{clf_name}-{self.cutoff_type}.png'),
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
        )
        plt.close()

    def visualize_scores(self, scores, y_test, train_scores,
            test_case_name, clf_name, RESULTS_DIR):
        p_vals = self.get_p_vals(scores)
        train_p_vals = self.get_p_vals(train_scores)

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

        os.makedirs(os.path.join(RESULTS_DIR, 'img', test_case_name), exist_ok=True)

        for metric in ['Score', 'p-value']:
            sns.set_theme()
            _, axs = plt.subplots(1, 2, figsize=(14, 6), 
                sharex=True, sharey=True)

            sns.histplot(train_df, x=metric, hue='Class', ax=axs[0],
                hue_order=['Inlier'], stat='probability')
            axs[0].set_title('Train')
            sns.histplot(df, x=metric, hue='Class', ax=axs[1],
                hue_order=['Inlier', 'Outlier'], stat='probability')
            axs[1].set_title('Test')
            
            plt.suptitle(f'{test_case_name} ({clf_name}, {self.cutoff_type}) - {metric} distribution')
            plt.savefig(
                os.path.join(RESULTS_DIR, 'img', test_case_name, f'distribution-{metric}-{clf_name}-{self.cutoff_type}.png'),
                dpi=150,
                bbox_inches='tight',
                facecolor='white',
            )
            plt.close()

    def visualize_roc(self, scores, y_test,
            test_case_name, clf_name, RESULTS_DIR):
        p_vals = self.get_p_vals(scores)
        
        inlier_idx = np.where(y_test == 1)[0]
        inlier_mask = np.isin(range(len(y_test)), inlier_idx)

        df = pd.DataFrame({
            'Score': scores,
            'p-value': p_vals,
            'Class': np.where(inlier_mask, 'Inlier', 'Outlier'),
        })

        os.makedirs(os.path.join(RESULTS_DIR, 'img', test_case_name), exist_ok=True)

        for metric in ['Score', 'p-value']:
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
                os.path.join(RESULTS_DIR, 'img', test_case_name, f'ROC-{metric}-{clf_name}-{self.cutoff_type}.png'),
                dpi=150,
                bbox_inches='tight',
                facecolor='white',
            )
            plt.close()

    def visualize_calibration(self, test_case_name, clf_name, RESULTS_DIR):
        N = len(self.X_train)
        cal_scores_all = np.zeros((self.resampling_repeats, N - int(N/2)))

        train_df = pd.DataFrame()

        mu_multis = []
        sigma_multis = []
        for i in range(self.resampling_repeats):
            multisplit_samples = np.random.choice(range(N), size=int(N/2), replace=False)
            is_multisplit_sample = np.isin(range(N), multisplit_samples)
            X_multi_train, X_multi_cal = self.X_train[is_multisplit_sample], self.X_train[~is_multisplit_sample]
            
            self.clf.fit(X_multi_train)
            cal_scores = self.clf.score_samples(X_multi_cal)
            cal_scores_all[i, :] = cal_scores

            train_df = pd.concat([train_df, pd.DataFrame({
                'Sample': np.char.add('C', (np.array(range(len(self.X_train))) + 1).astype(str)),
                'Split': np.char.add('Cal', np.repeat(i + 1, len(self.X_train)).astype(str)),
                'Score': self.clf.score_samples(self.X_train),
                'SampleType': np.where(is_multisplit_sample, 'Train', 'Calibration'),
            })])
            mu_multis.append(np.mean(X_multi_train, axis=0).reshape(1, -1))
            sigma_multis.append(np.cov(X_multi_train.T))

        mu_full = np.mean(self.X_train, axis=0).reshape(1, -1)
        sigma_full = np.cov(self.X_train.T)

        os.makedirs(os.path.join(RESULTS_DIR, 'img', test_case_name), exist_ok=True)

        self.clf.fit(self.X_train)
        train_scores = self.clf.score_samples(self.X_train)

        train_df = pd.concat([train_df, pd.DataFrame({
            'Sample': (np.array(range(len(self.X_train))) + 1).astype(str),
            'Split': 'Train',
            'Score': train_scores,
            'SampleType': 'Train',
        })])
        train_df = train_df.reset_index(drop=True)
        cal_df = train_df[(train_df.SampleType == 'Calibration') | (train_df.Split == 'Train')]

        sns.set_theme()

        fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
        fig.suptitle(f'{test_case_name} ({clf_name}) - scores per calibration split')

        ax = axs[0]
        sns.stripplot(data=cal_df, x='Score', y='Split', orient='h', ax=ax,
            hue='SampleType', hue_order=['Train', 'Calibration'],
            s=4, linewidth=0.2, alpha=0.7)
        ax.set_title(f'Calibration set only')

        ax.get_legend().remove()

        ax = axs[1]
        sns.stripplot(data=train_df, x='Score', y='Split', orient='h', ax=ax,
            hue='SampleType', hue_order=['Train', 'Calibration'],
            s=4, linewidth=0.2, alpha=0.7)
        ax.set_title(f'{test_case_name} ({clf_name}) - scores per calibration split, whole train dataset')
        ax.set_title(f'Whole train')

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center')
        ax.get_legend().remove()

        plt.tight_layout()
        plt.savefig(
            os.path.join(RESULTS_DIR, 'img', test_case_name, f'diagnostic-scores-{clf_name}-{self.cutoff_type}.png'),
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
        )
        plt.close(fig)

        if clf_name == 'Mahalanobis':
            conf_matrix_sigma = np.eye(self.resampling_repeats + 1)
            for i, sim1 in enumerate(sigma_multis + [sigma_full]):
                for j, sim2 in enumerate(sigma_multis + [sigma_full]):
                    conf_matrix_sigma[i, j] = np.linalg.norm(sim1 - sim2, 'fro')
            
            conf_matrix_mu = np.eye(self.resampling_repeats + 1)
            for i, sim1 in enumerate(mu_multis + [mu_full]):
                for j, sim2 in enumerate(mu_multis + [mu_full]):
                    conf_matrix_mu[i, j] = np.linalg.norm(sim1 - sim2, 2)

            labels = list(np.char.add('Cal', (np.array(range(self.resampling_repeats)) + 1).astype(str)))\
                + ['Train']

            sns.reset_defaults()

            fig, axs = plt.subplots(1, 2, figsize=(20, 8))
            fig.suptitle(f'{test_case_name} - pairwise distances between {clf_name} params')

            ax = axs[0]
            disp = metrics.ConfusionMatrixDisplay(
                confusion_matrix=conf_matrix_sigma,
                display_labels=labels,
            )
            disp.plot(ax=ax)
            ax.set_title(f'Covariance matrices (Frobenius norm)')

            ax = axs[1]
            disp = metrics.ConfusionMatrixDisplay(
                confusion_matrix=conf_matrix_mu,
                display_labels=labels,
            )
            disp.plot(ax=ax)
            ax.set_title(f'Mean vectors')

            plt.savefig(
                os.path.join(RESULTS_DIR, 'img', test_case_name, f'diagnostic-params-{clf_name}-{self.cutoff_type}.png'),
                dpi=150,
                bbox_inches='tight',
                facecolor='white',
            )
            plt.close(fig)

        return cal_scores_all

class PValueCutoff(Cutoff):
    base_cutoff: Cutoff

    @property
    def full_cutoff_type(self):
        return self.base_cutoff.cutoff_type + '+' + self.cutoff_type
    
    @property
    def short_cutoff_type(self):
        raise NotImplementedError('Use derived class for cutoff')
    
    def __init__(self, base_cutoff):
        # base cutoff needs to be fitted already
        self.base_cutoff = base_cutoff
        assert hasattr(self.base_cutoff, 'get_p_vals')
        assert hasattr(self.base_cutoff, 'is_fitted') and self.base_cutoff.is_fitted

    def apply(self, scores):
        p_vals = self.base_cutoff.get_p_vals(scores)
        y_pred = self.apply_to_pvals(p_vals)
        return y_pred

    def apply_to_pvals(self, p_vals):
        raise NotImplementedError('Use derived class for cutoff')

    def _get_thresholds(self, N):
        raise NotImplementedError('Use derived class for cutoff')
    
    def _calculate_threshold(self, sorted_p_vals, train_thresholds):
        raise NotImplementedError('Use derived class for cutoff')
    
    def fit(self, scores):
        p_vals = self.base_cutoff.get_p_vals(scores)

        p_vals = np.sort(p_vals)
        train_thresholds = self._get_thresholds(len(p_vals))
        self.threshold_ = self._calculate_threshold(p_vals, train_thresholds)

        self.is_fitted = True
        return self.threshold_

    def apply_to_pvals(self, p_vals):
        y_pred = np.where(p_vals < self.threshold_, 0, 1)
        return y_pred

    def visualize(self, scores, y_test, figure, \
            test_case_name, clf_name, RESULTS_DIR, 
            zoom=False, zoom_left=True, save_plot=False):
        assert self.is_fitted, 'Cutoff needs to be fitter first'
        
        N = len(scores)

        p_vals = self.base_cutoff.get_p_vals(scores)
        thresholds = self._get_thresholds(N)

        p_val_order = np.argsort(p_vals)
        p_vals, y_test = p_vals[p_val_order], y_test[p_val_order] # sort

        y_pred = np.zeros_like(p_vals)

        p_vals_below_bar = np.where(p_vals < thresholds)[0]
        if len(p_vals_below_bar) > 0:
            first_fulfillment_index = p_vals_below_bar[0]
            y_pred[first_fulfillment_index:] = 1

        x = np.array(range(N)) / N
        outlier_vec = np.where(y_test == 0, 'Outlier', 'Inlier')
        rejected_vec = np.where(y_pred == 0, 'Rejected', 'Not rejected')
        
        num_elements = N
        if zoom:
            num_elements = 100
            if zoom_left:
                if np.sum(y_pred == 0) < len(y_pred):
                    num_elements = max(num_elements, int(1.5 * np.sum(y_pred == 0)))
                num_elements = min(N, num_elements)

                x, p_vals, outlier_vec, rejected_vec, thresholds = \
                    tuple([a[:num_elements] \
                        for a in [x, p_vals, outlier_vec, rejected_vec, thresholds]
                    ])
            else:
                if np.sum(y_pred == 0) < len(y_pred):
                    num_elements = max(num_elements, int(1.5 * np.sum(y_pred == 1)))
                num_elements = min(N, num_elements)

                x, p_vals, outlier_vec, rejected_vec, thresholds = \
                    tuple([a[-num_elements:] \
                        for a in [x, p_vals, outlier_vec, rejected_vec, thresholds]
                    ])

        fig, ax = figure
        sns.scatterplot(x=x, y=p_vals, 
            hue=outlier_vec, style=rejected_vec,
            hue_order=['Inlier', 'Outlier'],
            style_order=['Not rejected', 'Rejected'],
            edgecolor='k', linewidth=.2,
            ax=ax)
        sns.scatterplot(x=x, y=thresholds, 
            hue=['Threshold'] * num_elements,
            palette=['r'],
            edgecolor=None, s=2,
            ax=ax)
        
        ax.set_xlabel('p-value quantile')
        ax.set_ylabel('Value')
        ax.set_ylim(-0.01, 1.01)
        
        legend = ax.legend()
        legend.legendHandles[-1]._sizes = [8.]

        ax.set_title(f'{self.cutoff_type}, alpha={self.alpha:.3f}' + \
            f', alpha={self.alpha:.3f}' + \
            f'{f" (zoomed)" if zoom else ""}')

        if save_plot:
            fig.tight_layout()
            fig.savefig(
                os.path.join(RESULTS_DIR, 'img', test_case_name, 
                    f'{self.short_cutoff_type}-{clf_name}-{self.base_cutoff.cutoff_type}.png'),
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
            )
            plt.close(fig)


class FORControlCutoff(PValueCutoff):
    @property
    def cutoff_type(self):
        return 'FOR-CTL'
    
    @property
    def short_cutoff_type(self):
        return 'FOR-CTL'

    def __init__(self, base_cutoff, alpha, pi):
        super().__init__(base_cutoff)
        self.alpha = alpha
        self.pi = pi
    
    def _get_thresholds(self, N):
        i_array = np.array(range(N))
        return 1 - ((1 - (i_array / N)) * (1 - self.alpha)) / (self.pi)

    def _calculate_threshold(self, sorted_p_vals, train_thresholds):
        p_vals_below_bar = np.where(sorted_p_vals < train_thresholds)[0]
        if len(p_vals_below_bar) > 0:
            first_fulfillment_index = p_vals_below_bar[0]
            return sorted_p_vals[first_fulfillment_index]
        else:
            return np.inf


class FNRControlCutoff(PValueCutoff):
    @property
    def cutoff_type(self):
        return 'FNR-CTL'
    
    @property
    def short_cutoff_type(self):
        return 'FNR-CTL'

    def __init__(self, base_cutoff, alpha, pi):
        super().__init__(base_cutoff)
        self.alpha = alpha
        self.pi = pi
    
    def _get_thresholds(self, N):
        i_array = np.array(range(N))
        return 1 - ((1 - (i_array / N)) - self.alpha * (1 - self.pi)) / (self.pi)

    def _calculate_threshold(self, sorted_p_vals, train_thresholds):
        p_vals_below_bar = np.where(sorted_p_vals < train_thresholds)[0]
        if len(p_vals_below_bar) > 0:
            first_fulfillment_index = p_vals_below_bar[0]
            return sorted_p_vals[first_fulfillment_index]
        else:
            return np.inf


class CombinedFORFNRControlCutoff(PValueCutoff):
    @property
    def cutoff_type(self):
        return 'FOR-FNR-CTL'
    
    @property
    def short_cutoff_type(self):
        return 'FOR-FNR-CTL'

    def __init__(self, base_cutoff, alpha, pi):
        super().__init__(base_cutoff)
        self.alpha = alpha
        self.pi = pi
    
    def _get_thresholds(self, N):
        i_array = np.array(range(N))
        return np.minimum( # use more tight one
            1 - ((1 - (i_array / N)) * (1 - self.alpha)) / (self.pi), # FOR
            1 - ((1 - (i_array / N) - self.alpha * (1 - self.pi))) / (self.pi), # FNR
        )

    def _calculate_threshold(self, sorted_p_vals, train_thresholds):
        p_vals_below_bar = np.where(sorted_p_vals < train_thresholds)[0]
        if len(p_vals_below_bar) > 0:
            first_fulfillment_index = p_vals_below_bar[0]
            return sorted_p_vals[first_fulfillment_index]
        else:
            return np.inf


class BenjaminiHochbergCutoff(PValueCutoff):
    @property
    def cutoff_type(self):
        return 'BH' \
            + (f'+pi' if self.pi is not None!= 1 else '')
    
    @property
    def short_cutoff_type(self):
        return 'BH'
    
    def __init__(self, base_cutoff, alpha, pi=None):
        super().__init__(base_cutoff)
        self.pi = pi

        if self.pi is None:
            self.alpha = alpha
        else:
            self.alpha = alpha / self.pi

    def _get_thresholds(self, N):
        i_array = np.array(range(N))        
        return (i_array / N) * self.alpha

    def _calculate_threshold(self, sorted_p_vals, train_thresholds):
        p_vals_below_bar = np.where(sorted_p_vals < train_thresholds)[0]
        if len(p_vals_below_bar) > 0:
            last_fulfillment_index = p_vals_below_bar[-1]
            return sorted_p_vals[last_fulfillment_index]
        else:
            return 0


# class FORControlObjectiveCutoff(__FORControlCutoffBase):
#     def __init__(self, inlier_rate, alpha):
#         super().__init__(inlier_rate, alpha)
    
#     def _for_estimate(self, test_ecdf, threshold):
#         return max(0, (1 - test_ecdf(threshold)) - self.inlier_rate * (1 - threshold)) / (1 - self.inlier_rate)
    
#     def _fnr_estimate(self, test_ecdf, threshold):
#         return max(0, (1 - test_ecdf(threshold)) - self.inlier_rate * (1 - threshold)) / (1 - test_ecdf(threshold))\
#             if test_ecdf(threshold) < 1 else 1 # technically 0, but undesired

#     def __objective_small(test_ecdf, threshold):
#         return (np.abs(threshold)) / 100

#     def __objective_for(self, test_ecdf, threshold):
#         for_est = self.__for_estimate(test_ecdf, threshold)
#         return np.where(
#                     for_est > self.alpha,
#                     ((for_est) / self.alpha) ** 2 - 1,
#                     0
#                 )

#     def __objective_fnr(self, test_ecdf, threshold):
#         fnr_est = self.__fnr_estimate(test_ecdf, threshold)
#         return np.where(
#                     fnr_est > self.alpha,
#                     ((fnr_est) / self.alpha) ** 2 - 1,
#                     0
#                 )
    
#     def __objective(self, test_ecdf, threshold):
#         return self.__objective_small(test_ecdf, threshold) \
#             + self.__objective_for(test_ecdf, threshold) \
#             + self.__objective_fnr(test_ecdf, threshold)

#     def fit(self, p_vals):
#         super().check_input(p_vals)
#         test_ecdf = ECDF(p_vals)

#         objective_values = []
#         all_possible_ecdf_values = np.linspace(0, 1, len(p_vals) + 1)
#         for threshold in all_possible_ecdf_values:
#             objective_values.append(self.__objective(test_ecdf, threshold))
        
#         objective_values = np.array(objective_values)
#         best_val_idx = np.argmin(objective_values)
        
#         self.threshold_ = all_possible_ecdf_values[best_val_idx]
#         return self.threshold_

#     def apply(self, p_vals):
#         y_pred = np.where(p_vals <= self.threshold_, 0, 1)
#         return y_pred
    
#     # def visualize(self, p_vals):
#     #     test_ecdf = ECDF(p_vals)

#     #     objective_values = []
#     #     objective_small_values = []
#     #     objective_for_values = []
#     #     objective_fnr_values = []

#     #     all_possible_ecdf_values = np.linspace(0, 1, len(p_vals) + 1)
#     #     for threshold in all_possible_ecdf_values:
#     #         objective_values.append(self.__objective(test_ecdf, threshold))
#     #         objective_small_values.append(self.objective_small(test_ecdf, threshold))
#     #         objective_for_values.append(self.objective_for(test_ecdf, threshold))
#     #         objective_fnr_values.append(self.objective_fnr(test_ecdf, threshold))

#     #     plt.figure(figsize=(8, 6))
#     #     plt.plot(all_possible_ecdf_values, objective_values)
#     #     plt.plot(all_possible_ecdf_values, objective_for_values)
#     #     plt.plot(all_possible_ecdf_values, objective_fnr_values)
#     #     plt.plot(all_possible_ecdf_values, objective_small_values)
        
#     #     plt.legend(['Full', '1', '2', '5'])
#     #     plt.ylabel('Objective values')
#     #     plt.xlabel('Threshold')
#     #     plt.savefig(f'{dataset}.png', dpi=300)
#     #     plt.show()
#     #     plt.close()
