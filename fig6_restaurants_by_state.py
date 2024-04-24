import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import cast_int, get_states_within_range
from dp_utils import re_estimate_tot_visits_by_column, re_estimate_avg_duration_by_column

colors = ['#e72a8a', '#e6ab01', '#7570b3', '#1c9e77', '#d95f02', '#67a61f']


def capitalize_state_names(states: list):
    return [" ".join([word.capitalize() for word in state.split(" ")]) for state in states]


def plot_visit_variation_by_state(
        df: pd.DataFrame, 
        eps: float, 
        keep_states: list, 
        categories: list, 
        normalize: bool=False
        ):
    
    eps_list = [eps, np.inf]
    diff_list = []
    for eps in eps_list:
        dft = df[(df["eps"] == eps)].reset_index(drop=True)

        before = dft[dft["difference_from_mandate"] < 0].reset_index(drop=True)
        before = before.groupby(["state", "top_category"])[["tot_visits", "tot_min_dwell"]].sum().reset_index(drop=False)

        before = re_estimate_tot_visits_by_column(before, eps, aggr_by='state')

        after = dft[dft["difference_from_mandate"] > 0].reset_index(drop=True)
        after = after.groupby(["state", "top_category"])[["tot_visits", "tot_min_dwell"]].sum().reset_index(drop=False)

        after = re_estimate_tot_visits_by_column(after, eps, aggr_by='state')

        diff = before.copy()
        diff['pred_est'] = after['pred_est'] - before['pred_est']
        if normalize:
            diff['pred_est'] = (after['pred_est'] - before['pred_est']) / before['pred_est']

        diff_list.append(diff)


    for category in np.random.choice(categories, 1, replace=False):

        plt.figure()
        x_range = np.arange(len(keep_states))

        width = 1/(len(eps_list)+1)
        for i, (eps, diff) in enumerate(zip(eps_list, diff_list)):
            y = diff[diff['top_category']==category]['pred_est']
            shift = (i-len(eps_list)/2)*width

            plt.bar(x_range+shift, y, width, color=colors[i], label=f'epsilon_c={cast_int(eps/2)}')
        plt.xticks(x_range, capitalize_state_names(keep_states), rotation=90)

        plt.title(f"{category} (Total visits)")
        plt.tight_layout()

        plt.legend()
        plt.draw()


def plot_duration_variation_by_state(
        df: pd.DataFrame, 
        eps: float, 
        keep_states: list, 
        categories: list, 
        normalize: bool=False
    ):

    # No LDP
    dft = df[(df["eps"] == np.inf)].reset_index(drop=True)
    before = dft[dft["difference_from_mandate"] < 0].reset_index(drop=True)
    before = before.groupby(["state", "top_category"])[["tot_visits", "tot_min_dwell"]].sum().reset_index(drop=False)
    before['truth'] = before['tot_min_dwell']/before['tot_visits']
    after = dft[dft["difference_from_mandate"] > 0].reset_index(drop=True)
    after = after.groupby(["state", "top_category"])[["tot_visits", "tot_min_dwell"]].sum().reset_index(drop=False)
    after['truth'] = after['tot_min_dwell']/after['tot_visits']

    diff = before.copy()
    diff['truth'] = after['truth'] - before['truth']
    if normalize:
        diff['truth'] = (after['truth'] - before['truth']) / before['truth']

    # LDP (No re-estimation)
    dft = df[(df["eps"] == eps)].reset_index(drop=True)
    before = dft[dft["difference_from_mandate"] < 0].reset_index(drop=True)
    before = before.groupby(["state", "top_category"])[["tot_visits", "tot_min_dwell"]].sum().reset_index(drop=False)
    before['pred_no_est'] = before['tot_min_dwell']/before['tot_visits']
    before['pred_est'] = re_estimate_avg_duration_by_column(before, eps, aggr_by='state')['pred_est']

    after = dft[dft["difference_from_mandate"] > 0].reset_index(drop=True)
    after = after.groupby(["state", "top_category"])[["tot_visits", "tot_min_dwell"]].sum().reset_index(drop=False)
    after['pred_no_est'] = after['tot_min_dwell']/after['tot_visits']
    after['pred_est'] = re_estimate_avg_duration_by_column(after, eps, aggr_by='state')['pred_est']

    diff['pred_no_est'] = after['pred_no_est'] - before['pred_no_est']
    diff['pred_est'] = after['pred_est'] - before['pred_est']
    if normalize:
        diff['pred_no_est'] = (after['pred_no_est'] - before['pred_no_est']) / before['pred_no_est']
        diff['pred_est'] = (after['pred_est'] - before['pred_est']) / before['pred_est']


    for category in np.random.choice(categories, 1, replace=False):

        plt.figure()

        x_range = np.arange(len(keep_states))

        labels = [f'epsilon={eps} (re-estimation)',f'epsilon={eps} (no re-estimation)', f'epsilon={np.inf}']
 
        width = 0.25

        y0 = diff[diff['top_category']==category]['pred_est']
        y1 = diff[diff['top_category']==category]['pred_no_est']
        y2 = diff[diff['top_category']==category]['truth']
        
        shift = -0.66
        plt.bar(x_range+shift, y0, width, color=colors[0], label=labels[0])
        shift += 0.25
        plt.bar(x_range+shift, y1, width, color=colors[1], label=labels[1])
        shift += 0.25
        plt.bar(x_range+shift, y2, width, color=colors[2], label=labels[2])
        plt.xticks(x_range, capitalize_state_names(keep_states), rotation=90)
        plt.title(f"{category} (Average visit duration)")
        plt.tight_layout()
        plt.legend()
        plt.draw()


if __name__ == "__main__":

    normalize = False
    categories = ['Restaurants and Other Eating Places']
    delta_days = 14

    df = pd.read_csv("metadata/combined.csv")

    day_range = np.arange(-delta_days,delta_days+1)
    df = df[df["difference_from_mandate"].isin(day_range)].reset_index(drop=True) 
    keep_states = get_states_within_range(df)
    df = df[df["state"].isin(keep_states)].reset_index(drop=True)

    grouped_df = df.groupby(["eps", "difference_from_mandate", "top_category", "state"])[["tot_visits", "tot_min_dwell"]].sum()
    df = grouped_df.reset_index()

    plot_visit_variation_by_state(df, 1, keep_states, categories)    
    plot_duration_variation_by_state(df, 4, keep_states, categories)
    
    plt.show()