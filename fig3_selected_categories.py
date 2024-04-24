import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dp_utils import re_estimate_tot_visits_by_day
from utils import cast_int

colors = ['C0', 'tab:grey', 'tab:pink']
styles = ['s-', 'o--', '^--']


def plot_multiple_epsilon(df: pd.DataFrame, top_categories: list, epsilon_list):
    unique_top_categories = df["top_category"].unique()
    df = df[df["eps"].isin(epsilon_list)]
    df = df.reset_index(drop=True)
    # Keep the necessary dates - May 1 - May 29
    dates = pd.date_range('2020-05-01', periods = 30, freq = 'D')
    df["local_date"] = pd.to_datetime(df["local_date"])
    df = df[df["local_date"].isin(dates)].reset_index(drop=True)

    plt.figure(figsize=(14, 6))

    x_ticks = dates.strftime('%d')
    x_range = np.arange(len(dates))

    for j, eps in enumerate(epsilon_list):
        temp_df = df[df["eps"] == eps]
        temp_df.loc[:, "tot_visits"] = re_estimate_tot_visits_by_day(temp_df, eps/2, unique_top_categories)
        temp_df = temp_df[temp_df["top_category"].isin(top_categories)].reset_index(drop=True)
        for i, category in enumerate(top_categories):
            temp_temp_df = temp_df[temp_df["top_category"] == category].reset_index(drop=True)
            plt.plot(x_range, temp_temp_df["tot_visits"], styles[j], color = colors[i], label = f"{category}, eps_c = {cast_int(eps/2)}")
    plt.xlabel('Day (May, 2020)', fontsize=7)
    plt.ylabel('Total Number of visits', fontsize=7)
    plt.xlim([x_range[0]-0.1, x_range[-1]+0.1])
    plt.xticks(x_range, x_ticks)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    plt.grid(linestyle=':')
    plt.legend(fontsize=7)
    plt.draw()


if __name__ == "__main__":

    df = pd.read_csv("metadata/combined.csv")

    grouped_df = df.groupby(["eps", "local_date", "top_category"])["tot_visits"].sum()
    df = grouped_df.reset_index()

    categories = ['Restaurants and Other Eating Places', 'Museums, Historical Sites, and Similar Institutions',
                  'Other Amusement and Recreation Industries']

    epsilon = [np.inf, 1]
    plot_multiple_epsilon(df, categories, epsilon)
    plt.show()
