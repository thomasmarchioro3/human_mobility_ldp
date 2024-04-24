import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dp_utils import re_estimate_avg_duration_histogram
from utils import get_states_within_range

def plot_key_event_duration(df: pd.DataFrame):
    eps_list = df['eps'].unique()

    accuracy_list = []

    temp_df = df.copy()
    temp_df = df[df["eps"] == np.inf].reset_index(drop=True)
    before_truth = temp_df[temp_df["difference_from_mandate"] < 0].reset_index(drop=True)

    before_truth = before_truth.groupby("top_category")[["tot_visits","tot_min_dwell"]].sum().reset_index(drop=False)
    before_truth["avg_dwell"] = before_truth["tot_min_dwell"]/before_truth["tot_visits"]
    before_truth = before_truth.set_index("top_category")["avg_dwell"]

    after_truth = temp_df[temp_df["difference_from_mandate"] > 0].reset_index(drop=True)
    after_truth = after_truth.groupby("top_category")[["tot_visits","tot_min_dwell"]].sum().reset_index(drop=False)
    after_truth["avg_dwell"] = after_truth["tot_min_dwell"]/after_truth["tot_visits"]
    after_truth = after_truth.set_index("top_category")["avg_dwell"]

    ground_truth = after_truth > before_truth

    thresholds = [0, 0.5, 1]
    x_range = np.arange(len(eps_list))
    labels = ["No threshold", "30 seconds", "1 minute"]
    i = 0
    plt.figure()

    for threshold in thresholds:
        accuracy_list = []
        for eps in eps_list:
            dft = df.copy()
            dft = df[df["eps"] == eps].reset_index(drop=True)

            before = dft[dft["difference_from_mandate"] < 0].reset_index(drop=True)
            before = before.groupby("top_category")[["tot_visits", "tot_min_dwell"]].sum().reset_index(drop=False)

            before['avg_duration_est'] = re_estimate_avg_duration_histogram(before, eps)
            before = before.set_index('top_category')['avg_duration_est']


            after = dft[dft["difference_from_mandate"] > 0].reset_index(drop=True)
            after = after.groupby("top_category")[["tot_visits", "tot_min_dwell"]].sum().reset_index(drop=False)
            after['avg_duration_est'] = re_estimate_avg_duration_histogram(after, eps)
            after = after.set_index('top_category')['avg_duration_est']

            ok_to_keep = ground_truth[abs(after_truth - before_truth) > threshold].index

            estimated = after > before


            estimated = estimated[ok_to_keep]

            ground_truth_ = ground_truth[ok_to_keep]

            count_correct = sum(estimated == ground_truth_)
            accuracy = count_correct/len(estimated)
            accuracy_list.append(accuracy)

        plt.plot(x_range, accuracy_list, 's-', label = labels[i])
        i += 1

    plt.xticks(x_range, eps_list)
    plt.xlim(x_range[0], x_range[-1])
    plt.ylim(0, 1.1)
    plt.xlabel('epsilon')
    plt.ylabel('accuracy (dwell)')
    plt.grid(linestyle=':')
    plt.legend()

    plt.show()



if __name__ == "__main__":

    df = pd.read_csv("metadata/combined.csv")

    delta_days = 14

    day_range = np.arange(-delta_days,delta_days+1)
    df = df[df["difference_from_mandate"].isin(day_range)].reset_index(drop=True) 
    keep_states = get_states_within_range(df)
    df = df[df["state"].isin(keep_states)].reset_index(drop=True)

    grouped_df = df.groupby(["eps", "difference_from_mandate", "top_category"])[["tot_visits", "tot_min_dwell"]].sum()
    df = grouped_df.reset_index()

    plot_key_event_duration(df)

    plt.show()
