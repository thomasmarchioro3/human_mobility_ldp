import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from dp_utils import cast_int, re_estimate_visits


def plot_error_visits(df, normalize=True):

    ground_truth = df[df["eps"] == np.inf].groupby(['top_category', 'local_date'])["tot_visits"].sum()
    ground_truth = ground_truth.sort_index().values # Sort based on category name to ensure order is correct

    eps_list = df["eps"].unique()
    maes = []
    rmses = []
    for eps in eps_list:
        temp_df = df[df["eps"] == eps].reset_index(drop=True)
        temp_df["tot_visits"] = re_estimate_visits(temp_df, eps/2)
        predicted = temp_df.groupby(['top_category', 'local_date'])['tot_visits'].sum()
        predicted = predicted.sort_index().values
        if normalize:
            mae = mean_absolute_error(np.ones_like(ground_truth), predicted/ground_truth)
            rmse = mean_squared_error(np.ones_like(ground_truth), predicted/ground_truth, squared=False)
        else:
            mae = mean_absolute_error(ground_truth, predicted)
            rmse = mean_squared_error(ground_truth, predicted, squared=False)
        maes.append(mae)
        rmses.append(rmse)
    print(rmses)
    plt.figure(figsize=(14, 6))
    x_range = np.arange(len(eps_list))
    plt.xticks(x_range, [cast_int(eps/2) for eps in eps_list])
    plt.plot(maes, 's-', label='MAE' if not normalize else 'NMAE', color='tab:blue')
    plt.plot(rmses, 's-', label='RMSE' if not normalize else 'NRMSE', color='tab:red')
    plt.xlim(x_range[0], x_range[-1])
    plt.grid(linestyle=':')
    plt.xlabel('epsilon_c')
    plt.ylabel('Error on daily visit estimation')
    plt.legend()

    plt.draw()

if __name__ == "__main__":



    normalize = False

    df = pd.read_csv("metadata/combined.csv")

    grouped_df = df.groupby(["eps", "local_date", "top_category"])["tot_visits"].sum()
    df = grouped_df.reset_index()

    plot_error_visits(df, normalize)
    plt.show()
