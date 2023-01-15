# %%
from occ_cutoffs import *
from occ_all_tests_lib import *
from IPython.display import display

class FakeClfTrain():
    def __init__(self, theta, pi, N=1_000):
        self.theta = theta
        self.pi = pi
        self.N = N

    def fit(self, X_train, y_train=None):
        pass

    def score_samples(self, X_test):
        inlier_scores = np.random.normal(0, 1, int(self.N))
        return inlier_scores

class FakeClfTest():
    def __init__(self, theta, pi, N=1_000):
        self.theta = theta
        self.pi = pi
        self.N = N

    def fit(self, X_train, y_train=None):
        pass

    def score_samples(self, X_test):
        inlier_scores = np.random.normal(0, 1, int(self.pi * self.N))
        outlier_scores = np.random.normal(-self.theta, 1, N - int(self.pi * self.N))
        return np.concatenate([inlier_scores, outlier_scores])

alpha = 0.05
N = 1_000
resampling_repeats = 10

for threshold in ['BH', 'BH+pi']:
    sns.set_theme()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca()

    for pi in np.linspace(0, 1, 11):
        fors = []
        thetas = np.linspace(0, 5, 51)
        for theta in thetas:
            y_test = np.concatenate([
                np.ones(int(pi * N)),
                np.zeros(N - int(pi * N)),
            ])
            # print(y_test)

            if pi == 0:
                fors.append(0)
                continue
            elif pi == 1:
                fors.append(1)
                continue

            construct_clf = lambda theta=theta, pi=pi, N=N: FakeClfTrain(theta, pi, N)

            X_train = np.zeros((N, 2))
            X_test = np.zeros((N, 2)) # dummies

            theta_fors = []
            for exp in range(10):
                cutoff = MultisplitCutoff(construct_clf, alpha=alpha, resampling_repeats=resampling_repeats)
                cutoff.fit(X_train)

                cutoff.set_clfs([
                    FakeClfTest(theta, pi, N)
                    for _ in range(resampling_repeats)
                ])
                
                if threshold == 'BH':
                    control_cutoff = BenjaminiHochbergCutoff(cutoff, alpha=alpha, pi=None)
                elif threshold == 'BH+pi':
                    control_cutoff = BenjaminiHochbergCutoff(cutoff, alpha=alpha, pi=pi)
                # control_cutoff = FORControlCutoffOld(cutoff, alpha=alpha, pi=pi)
                scores, y_pred = control_cutoff.fit_apply(X_test)
                
                DIR = os.path.join('plots')
                os.makedirs(DIR, exist_ok=True)
                
                # fig2 = plt.figure(figsize=(8, 6))
                # ax2 = plt.gca()

                # control_cutoff.visualize(X_test, y_test, (fig2, ax2), \
                #     f'{control_cutoff.cutoff_type} - FOR control ({pi=:.2f}, {theta=:.2f})', 
                #     'Fake classifier', DIR, 
                #     zoom=False, zoom_left=True, save_plot=True)

                method_metrics = prepare_metrics(y_test, y_pred, scores, {},
                    ['FOR', '#TP', '#FP', '#TN', '#FN'], pos_class_only=True)
                print(f'{pi=:.2f}, {theta=:.2f}, FOR: {method_metrics["FOR"]:.3f}')
                # display(method_metrics)
                theta_fors.append(method_metrics["FOR"])
            fors.append(np.mean(theta_fors))

        sns.lineplot(x=thetas, y=fors, label=f'$\\pi = {pi:.1f}$', ax=ax)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Distance between means $\\theta$')
    ax.set_ylabel('Simulated FOR')

    # ax.set_title(f'Simulated FOR value, FDR control ({threshold})')
    ax.legend()
    fig.savefig(os.path.join('plots', f'FOR_simulation_{threshold}.png'),
        bbox_inches='tight', facecolor='w', dpi=600)
    fig.savefig(os.path.join('plots', f'FOR_simulation_{threshold}.pdf'),
        bbox_inches='tight', facecolor='w')
    plt.show()
    plt.close(fig)


# class FakeCutoff(Cutoff):
#     is_fitted = True
#     cutoff_type = 'Fake'

#     def __init__(self, theta):
#         self.theta = theta
    
#     def fit(self, X_train, y_train=None):
#         pass
    
#     def apply(self, X_test, inlier_rate):
#         pass
    
#     def get_clfs(self):
#         return []

#     def get_p_vals(self, X_test):
#         return self.__get_nosplit_p_values(X_test)

#     def apply_to_p_vals(self, p_vals):
#         y_pred = np.where(p_vals < self.alpha, 0, 1)
#         return y_pred
    
#     def set_clfs(self, clfs):
#         pass

# %%
