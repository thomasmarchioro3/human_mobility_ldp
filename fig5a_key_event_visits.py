import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import cast_int, get_states_within_range

# from dp_utils import re_estimate_tot_visits_histogram as re_estimate_visits

def re_estimate_visits(dft, eps, categories=None):

    if categories is None:
        categories = dft["top_category"].unique()

    m = len(categories)
    exp_eps = np.exp(eps)
    p = 1/(1 + (exp_eps-1)/m)
    if p != 0:
        dft["tot_visits"] = (dft["tot_visits"] - p*dft["tot_visits"].sum()/m)/(1-p)
    dft.set_index("top_category", inplace=True)
    return dft['tot_visits']



def plot_key_event_visits(df):

    eps_list = df['eps'].unique()
    top_categories = df["top_category"].unique()

    accuracy_list = []

    temp_df = df.copy()
    temp_df = df[df["eps"] == np.inf].reset_index(drop=True)
    before_truth = temp_df[temp_df["difference_from_mandate"] < 0].reset_index(drop=True)
    before_truth = before_truth.groupby("top_category")["tot_visits"].sum()

    after_truth = temp_df[temp_df["difference_from_mandate"] > 0].reset_index(drop=True)
    after_truth = after_truth.groupby("top_category")["tot_visits"].sum()

    ground_truth = after_truth > before_truth

    for eps in eps_list:
        dft = df.copy()
        dft = df[df["eps"] == eps].reset_index(drop=True)

        before = dft[dft["difference_from_mandate"] < 0].reset_index(drop=True)
        before = before.groupby("top_category")["tot_visits"].sum().reset_index(drop=False)

        before = re_estimate_visits(before, eps)

        after = dft[dft["difference_from_mandate"] > 0].reset_index(drop=True)
        after = after.groupby("top_category")["tot_visits"].sum().reset_index(drop=False)

        after = re_estimate_visits(after, eps)

        estimated = after > before

        print(estimated)
        count_correct = sum(estimated == ground_truth)
        accuracy_list.append(count_correct/len(top_categories))

    x_range = np.arange(len(eps_list))

    plt.figure()
    plt.plot(x_range, accuracy_list, 's-')
    plt.xticks(x_range, [cast_int(eps/2) for eps in eps_list])
    plt.xlim(x_range[0], x_range[-1])
    plt.xlabel('epsilon_c')
    plt.ylabel('accuracy (visits)')
    plt.grid(linestyle=':')

    plt.draw()




if __name__ == "__main__":

    df = pd.read_csv("metadata/combined.csv")
    delta_days = 14

    day_range = np.arange(-delta_days,delta_days+1)
    df = df[df["difference_from_mandate"].isin(day_range)].reset_index(drop=True) 
    keep_states = get_states_within_range(df)
    df = df[df["state"].isin(keep_states)].reset_index(drop=True)

    grouped_df = df.groupby(["eps", "difference_from_mandate", "top_category"])["tot_visits"].sum()
    df = grouped_df.reset_index()

    plot_key_event_visits(df)

    plt.show()
