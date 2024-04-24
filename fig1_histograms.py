from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import cast_int
from utils import category_names

from dp_utils import re_estimate_tot_visits_histogram
from dp_utils import re_estimate_avg_duration_histogram

colors = ['#e72a8a', '#e6ab01', '#7570b3', '#1c9e77', '#d95f02', '#67a61f']

# def re_estimate_tot_visits(dft, eps):
#     top_categories = dft["top_category"].unique()
#     exp_eps = np.exp(eps/2)
#     p = 1/(1 + (exp_eps-1)/len(top_categories))

#     avg_visits_overall = dft["tot_visits"].mean()
#     tot_visits_est = (dft['tot_visits'] - p * avg_visits_overall)/(1-p)

#     return tot_visits_est

# def re_estimate_avg_duration(dft, eps):
#     top_categories = dft["top_category"].unique()
#     exp_eps = np.exp(eps/2)
#     p = 1/(1 + (exp_eps-1)/len(top_categories))

#     avg_visits_overall = dft["tot_visits"].mean()
#     avg_dwell_averall = dft['tot_min_dwell'].mean()
#     tot_visits_est = dft['tot_visits'] - p * avg_visits_overall#/(1-p)
#     tot_dwell_est = dft['tot_min_dwell'] - p * avg_dwell_averall

#     return tot_dwell_est / tot_visits_est

def plot_histogram_visits(df: pd.DataFrame, eps_list: Optional[list]=None):

    if eps_list is None:
        eps_list = [1, 2, np.inf]

    top_categories = df['top_category'].unique()
    top_categories = [category_names[category] for category in top_categories]

    plt.figure(figsize=(14, 8))
    width = 1/(len(eps_list)+1)
    
    

    x_range = np.arange(len(top_categories))

    for i, eps in enumerate(eps_list):
        shift = (i-len(eps_list)/2)*width
        estimated = re_estimate_tot_visits_histogram(
            df[df['eps'] == eps], 
            eps
            ).reset_index(drop=True)
        plt.bar(x_range+shift, estimated, width, color=colors[i], label=f'epsilon_c={cast_int(eps/2)}')
    plt.xticks(x_range, top_categories, rotation=90)
    plt.title('Number of visits')
    plt.tight_layout()

    plt.legend()
    plt.draw()


def plot_histogram_duration(df: pd.DataFrame, eps_list: Optional[list]=None):

    if eps_list is None:
        eps_list = [1, 2, np.inf]

    top_categories = df['top_category'].unique()
    top_categories = [category_names[category] for category in top_categories]
    x_range = np.arange(len(top_categories))

    plt.figure(figsize=(14, 8))
    width = 1/(len(eps_list)+1)    
    
    for i, eps in enumerate(eps_list):
        shift = (i-len(eps_list)/2)*width
        estimated = re_estimate_avg_duration_histogram(
            df[df['eps'] == eps], 
            eps
            ).reset_index(drop=True)
        plt.bar(x_range+shift, estimated, width, color=colors[i], label=f'epsilon={cast_int(eps)}')
    plt.xticks(x_range, top_categories, rotation=90)
    plt.tight_layout()
    plt.title('Average visit duration')
    plt.legend()
    plt.draw()



if __name__ == "__main__":

    eps_list = [1, 2, np.inf]
    df = pd.read_csv("metadata/combined.csv")

    grouped_df = df.groupby(["eps", "top_category"])[["tot_visits", "tot_min_dwell"]].sum()
    df = grouped_df.reset_index(drop=False)

    plot_histogram_visits(df, eps_list)
    plot_histogram_duration(df, eps_list)
    plt.show()
