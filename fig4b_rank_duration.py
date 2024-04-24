from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dp_utils import re_estimate_avg_duration_histogram


def plot_rank_error_duration(df: pd.DataFrame, day_column: Optional[str]=None):

    if day_column is None:
        day_column = 'local_date'

    days = df[day_column].unique()
    eps_list = df['eps'].unique()

    tot_correct = {eps: 0 for eps in eps_list}
    tot = {eps: 0 for eps in eps_list}

    eps_list = [eps for eps in eps_list if eps >= 1]

    for day in days:
        dft = df[df[day_column]==day].reset_index(drop=True)
        ground_truth = (dft[dft['eps']==np.inf]['tot_min_dwell']/dft[dft['eps']==np.inf]['tot_visits']).to_numpy()
        actual_diff = np.add.outer(ground_truth, -ground_truth)

        for eps in eps_list:
            estimated_counts = re_estimate_avg_duration_histogram(dft[dft['eps']==eps], eps).to_numpy()
            estimated_diff = np.add.outer(estimated_counts, -estimated_counts)  # all pairwise

            tot_correct[eps] += (actual_diff*estimated_diff > 0).sum()
            tot[eps] += np.prod(actual_diff.shape) - len(actual_diff)  # count all elements except the diagonal ones (avoid comparing a category with itself)

    error_list = [1-tot_correct[eps]/tot[eps] for eps in eps_list]
    x_range = np.arange(len(eps_list))

    plt.figure()
    plt.plot(x_range, error_list, 's-')
    plt.xticks(x_range, eps_list)
    plt.xlim(x_range[0], x_range[-1])
    plt.xlabel('epsilon')
    plt.ylabel('Error on pairwise comparison (dwell)')
    plt.grid(linestyle=':')
    plt.draw()



if __name__ == "__main__":


    df = pd.read_csv("metadata/combined.csv")

    grouped_df = df.groupby(["eps", "local_date", "top_category"])[["tot_visits", "tot_min_dwell"]].sum()
    df = grouped_df.reset_index()

    plot_rank_error_duration(df, day_column='local_date')

    plt.show()
