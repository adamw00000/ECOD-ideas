# %%
import os
import numpy as np
import pandas as pd
from occ_all_tests_lib import *

test_type = 'fdr'
DATASET_TYPE = 'BINARYdata'
alpha = 0.1

DIR = os.path.join('results', f'results_{DATASET_TYPE}_{test_type}_{alpha:.2f}')

df = pd.DataFrame()
for d in os.listdir(os.path.join(DIR)):
    dataset_dir = os.path.join(DIR, d)
    if not os.path.isdir(dataset_dir) or 'plots' in dataset_dir:
        continue

    for file in os.listdir(dataset_dir):
        if 'results-raw' in file and file.endswith('.csv'):
            df = pd.concat([
                df,
                pd.read_csv(os.path.join(dataset_dir, file))
            ])

df = df.reset_index(drop=True)
df = df[df.Dataset != 'Mean']
df[metrics_to_multiply_by_100] = df[metrics_to_multiply_by_100] / 100

metric_list = df.columns[3:]
alpha_metrics = [m for m in metric_list if 'alpha' in m]
metric_list = [m for m in metric_list if 'alpha' not in m]

df

# %%
import matplotlib.pyplot as plt
import seaborn as sns

cutoff = 'Multisplit'

def draw_plots(metric, cutoff_correction, alternative_metric):
    controlling_cutoff = f'{cutoff}+{cutoff_correction}'
    metric_df = df.loc[df.Cutoff == controlling_cutoff, ['Dataset', 'Method', 'Cutoff', metric, alternative_metric]]
    metric_df

    sns.set_theme()
    os.makedirs(os.path.join(DIR, 'global-plots'), exist_ok=True)

    for method in metric_df.Method.unique():
        method_df = metric_df[metric_df.Method == method]

        nan_metric_string = f'Undefined {metric} ({"no rejections" if metric == "FDR" else "all rejected"})'
        nan_alternative_string = f'Undefined {metric} ({"no rejections" if alternative_metric == "FDR" else "all rejected"})'
        normal_string = f'Defined {metric}'
        method_df = method_df.assign(
            IsNaN=np.where(np.isnan(method_df[metric]), nan_metric_string, normal_string),
            IsAlternativeNaN=np.where(np.isnan(method_df[alternative_metric]), nan_alternative_string, normal_string),
        )
        method_df = method_df.fillna(0)

        mean_df = method_df.groupby(['Dataset', 'Method', 'Cutoff']) \
            [[metric, alternative_metric]] \
            .mean() \
            .reset_index(drop=False)

        datasets = df.Dataset.unique()
        fig, axs = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
        strip = sns.stripplot(data=method_df, y='Dataset', x=metric, hue='IsNaN',
            edgecolor='k', alpha=0.6, linewidth=0.2,
            hue_order=[normal_string, nan_metric_string],
            orient='h',
            ax=axs[0],
        )
        strip.axvline(alpha, ls='--', c='k')
        sns.scatterplot(
            data=mean_df, y='Dataset', x=metric,
            hue=['Mean value'] * len(mean_df),
            style=['Mean value'] * len(mean_df),
            markers=['X'],
            palette=['r'],
            zorder=100, 
            edgecolor='k',
            s = 12,
            ax=axs[0],
        )
        axs[0].set_title(metric)

        strip = sns.stripplot(data=method_df, y='Dataset', x=alternative_metric, hue='IsAlternativeNaN',
            edgecolor='k', alpha=0.6, linewidth=0.2,
            hue_order=[normal_string, nan_alternative_string],
            orient='h',
            ax=axs[1],
        )
        strip.axvline(alpha, ls='--', c='k')
        sns.scatterplot(
            data=mean_df, y='Dataset', x=alternative_metric,
            hue=['Mean value'] * len(mean_df),
            style=['Mean value'] * len(mean_df),
            markers=['X'],
            palette=['r'],
            zorder=100, 
            edgecolor='k',
            s = 12,
            ax=axs[1],
        )
        axs[1].set_title(alternative_metric)

        fig.suptitle(f'{method} - {controlling_cutoff} ({metric})')
        axs[0].set_xlim(-0.01, metric_df[metric].max() + 0.01)
        axs[1].set_xlim(-0.01, metric_df[alternative_metric].max() + 0.01)

        fig.tight_layout()
        plt.savefig(
            os.path.join(DIR, 'global-plots', f'{cutoff_correction}-{method}.png'),
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
        )
        plt.savefig(
            os.path.join(DIR, 'global-plots', f'{cutoff_correction}-{method}.pdf'),
            bbox_inches='tight'
        )
        plt.show()
        plt.close(fig)

draw_plots('FDR', f'BH+pi', 'FOR')

draw_plots('FOR', 'FOR-CTL-old', 'FDR')
draw_plots('FOR', 'FOR-CTL-gt', 'FDR')
draw_plots('FOR', 'FOR-CTL-geq', 'FDR')
draw_plots('FOR', 'FOR-CTL-geq-accept-eq', 'FDR')

# %%
