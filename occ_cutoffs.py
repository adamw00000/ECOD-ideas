import numpy as np
import scipy.stats

class Cutoff():
    @property
    def cutoff_type(self):
        raise NotImplementedError('Use derived class for cutoff')

    def fit(self, scores) -> float:
        raise NotImplementedError('Use derived class for cutoff')
    
    def apply(self, scores):
        raise NotImplementedError('Use derived class for cutoff')

    def fit_apply(self, scores):
        raise NotImplementedError('Use derived class for cutoff')


class EmpiricalCutoff(Cutoff):
    cutoff_type = 'Empirical'

    def __init__(self, inlier_rate):
        self.inlier_rate = inlier_rate

    def fit(self, scores):
        self.threshold_ = np.quantile(scores, q=1 - self.inlier_rate)
        return self.threshold_
    
    def apply(self, scores):
        y_pred = np.where(scores > self.threshold_, 1, 0)
        return y_pred

    def fit_apply(self, scores):
        self.fit(scores)
        return self.apply(scores)


class ChiSquaredCutoff(Cutoff):
    cutoff_type = 'Chi-squared'

    def __init__(self, inlier_rate, d):
        self.inlier_rate = inlier_rate
        # d - number of features
        self.d = d

    def fit(self, scores):
        self.threshold_ = -scipy.stats.chi2.ppf(1 - self.inlier_rate, 2 * self.d)
        return self.threshold_
    
    def apply(self, scores):
        y_pred = np.where(scores > self.threshold_, 1, 0)
        return y_pred

    def fit_apply(self, scores):
        self.fit(scores)
        return self.apply(scores)


class __ResamplingThresholdCutoff(Cutoff):
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
        return self.threshold_

    def apply(self, scores):
        y_pred = np.where(scores > self.threshold_, 1, 0)
        return y_pred

    def fit_apply(self, scores):
        self.fit(scores)
        return self.apply(scores)


class BootstrapThresholdCutoff(__ResamplingThresholdCutoff):
    cutoff_type = 'Bootstrap_threshold'

    def __init__(self, inlier_rate, resampling_repeats, X_train, clf):
        super().__init__(inlier_rate, resampling_repeats, X_train, clf)

    def _choose_samples(self, N):
        return np.random.choice(range(N), size=N, replace=True)

class MultisplitThresholdCutoff(__ResamplingThresholdCutoff):
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
        return self.cal_scores_

    def get_p_vals(self, scores):
        return self.__get_multisplit_p_values(scores)
    
    def apply_to_p_vals(self, p_vals):
        y_pred = np.where(p_vals < self.alpha, 0, 1)
        return y_pred
    
    def apply(self, scores):
        p_vals = self.__get_multisplit_p_values(scores)
        return self.apply_to_p_vals(p_vals)

    def fit_apply(self, scores):
        self.fit(scores)
        return self.apply(scores)