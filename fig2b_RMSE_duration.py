from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from dp_utils import re_estimate_avg_duration_by_day
from utils import cast_int

colors = ['tab:red']+[f'C{i}' for i in range(10)]


def get_estimated_durations(df: pd.DataFrame, eps: float):

    df_true = df[df["eps"] == np.inf].copy().reset_index(drop=True)

    results = df.groupby(['local_date', 'top_category'])['tot_visits'].sum().reset_index(drop=False)

    ground_truth = df_true.groupby(['top_category', 'local_date']).apply(
        lambda x: x.tot_min_dwell.sum() / x.tot_visits.sum(), include_groups=False
    ).reset_index(drop=False).rename(columns={0:'ground_truth'})

    results = results.merge(ground_truth, how='left', on=['top_category', 'local_date'])

    df_noisy = df[df["eps"] == eps].copy().reset_index(drop=True)

    pred_no_est = df_noisy.groupby(['top_category', 'local_date']).apply(
        lambda x: x.tot_min_dwell.sum() / x.tot_visits.sum(), include_groups=False
    ).reset_index(drop=False).rename(columns={0:'pred_no_est'})

    results = results.merge(pred_no_est, how='left', on=['top_category', 'local_date'])

    pred_est = re_estimate_avg_duration_by_day(df_noisy, eps)

    results = results.merge(pred_est, how='left', on=['top_category', 'local_date'])

    return results


def plot_dwell_error(df, metrics: Optional[list]=None, normalize=False, eps_list=None, visit_threshold=0, plot_no_est=False):

    if metrics is None:
        metrics = ['mae']

    if eps_list is None:
        eps_list = [cast_int(eps) for eps in df['eps'].unique() if eps != np.inf]

    error_no_est = {metric: [] for metric in metrics}
    error_est = {metric: [] for metric in metrics}

    for eps in eps_list:

        results = get_estimated_durations(df, eps)

        if visit_threshold > 0:
            results = results[results['tot_visits'] > visit_threshold]

        for metric in metrics:

            error_fn = lambda x, y: mean_absolute_error(x, y)  # mae by default
            if metric.lower() == 'rmse':
                error_fn = lambda x, y: mean_squared_error(x, y, squared=False)
            elif metric.lower() == 'mse':
                error_fn = lambda x, y: mean_squared_error(x, y, squared=True)
            elif metric.lower() == 'me':
                error_fn = lambda x, y: np.mean(x-y)

            if normalize:
                error_no_est[metric].append(error_fn(np.ones(len(results)), results['pred_no_est']/results['ground_truth']))
                error_est[metric].append(error_fn(np.ones(len(results)), results['pred_est']/results['ground_truth']))
            else:
                error_no_est[metric].append(error_fn(results['ground_truth'], results['pred_no_est']))
                error_est[metric].append(error_fn(results['ground_truth'], results['pred_est']))

    x_range = np.arange(len(eps_list))

    

    plt.figure(figsize=(14, 6))
    for i, metric in enumerate(metrics):
        plt.semilogy(x_range, error_est[metric], 's-', label=f'{metric.upper()}', color=colors[i])
        if plot_no_est:
            plt.plot(x_range, error_no_est, 's--', label=f'{metric.upper()} (no re-estimation)', color=colors[i])
    plt.xlim([x_range[0],x_range[-1]])
    plt.xticks(x_range, eps_list)
    plt.grid(linestyle=':')
    plt.ylabel(f'{"N" if normalize else ""} Error on dwell estimation (log)')
    plt.xlabel('Epsilon')
    plt.legend()

    plt.draw()
    plt.show()


if __name__ == "__main__":

    metrics=['rmse', 'mae']
    normalize=False
    visit_threshold = 0

    # NOTE: Uncomment the next line to get the "highly visited" curves
    # visit_threshold = 5e6
    
    eps_list = None

    df = pd.read_csv("metadata/combined.csv")

    grouped_df = df.groupby(["eps", "local_date", "top_category"])[["tot_visits", "tot_min_dwell"]].sum()
    df = grouped_df.reset_index(drop=False)

    plot_dwell_error(df, metrics, normalize, eps_list, visit_threshold=visit_threshold)

