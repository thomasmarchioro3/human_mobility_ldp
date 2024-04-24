from typing import Optional

import numpy as np
import pandas as pd


def randomized_response(df: pd.DataFrame, column: str, eps: float, values: list):
    """
    Applies generalized randomized response on the specified (categorical) column of the input dataframe to achieve LDP.
    
    Args:
        df (pd.DataFrame): The dataframe to be randomized
        column (str): The name of the column to be randomized
        values (1D ArrayLike): Possible values of the column
        eps (float): The privacy budget
    """

    exp_eps = np.exp(eps)
    p = 1/(1 + (exp_eps-1)/len(values))
    rx = np.random.choice(values, size=len(df)) 
    coin_flips = np.random.rand(len(df))
    df.loc[coin_flips < p, column] = rx[coin_flips < p]
    return df


def laplace_randomizer(df: pd.DataFrame, column: list, eps: float, min_val: float, max_val: float):
    """
    Applies the Laplace mechanism on the specified (numeric) column of the input dataframe to achieve LDP.
    
    Args:
        df (pd.DataFrame): The dataframe to be randomized
        column (str): The name of the column to be randomized
        eps (float): The privacy budget
        min_val
    """
    df.loc[df[column] < min_val, column] = min_val
    df.loc[df[column] > max_val, column] = max_val
    scale = (max_val - min_val) / eps
    df[column] = df[column] + np.random.laplace(loc=0, scale=scale, size=len(df))
    return df
    

def randomize_dataset(df: pd.DataFrame, eps, top_categories, max_duration=240):  
    """
    Randomize the dataset to achieve eps-LDP.

    Args:
        df (pd.DataFrame): The dataframe to be randomized
        eps (float): The privacy budget
        top_categories (1D ArrayLike): List of top categories to be kept in the randomized dataset. Others are discarded.
        max_duration: Maximum value for the visit duration. Visit duration values are clipped to `max_duration`.
    """

    df = df.copy()
    df = df[df['top_category'].isin(top_categories)].reset_index(drop=True)
    df = randomized_response(df, 'top_category', eps/2, top_categories)
    df = laplace_randomizer(df, 'minimum_dwell', eps/2, 0, max_duration)
    return df

def randomize_dataset_multidwells(df: pd.DataFrame, eps: float, top_categories: list, max_duration_list: Optional[list]=None):

    if max_duration_list is None:
        max_duration_list = [120, 240]

    df = df.copy()
    df = randomized_response(df, 'top_category', eps/2, top_categories)
    for i, max_duration in enumerate(max_duration_list):
        df[f'minimum_dwell_{i+1}'] = df['minimum_dwell']
        df = laplace_randomizer(df, f'minimum_dwell_{i+1}', eps/2, 0, max_duration)
    return df


def re_estimate_tot_visits_by_day(df: pd.DataFrame, eps: float, categories: Optional[list]=None):
    
    if categories is None:
        categories = df["top_category"].unique()
    
    m = len(categories)
    exp_eps = np.exp(eps)
    p = 1/(1 + (exp_eps-1)/m)
    if p == 0:
        return df['tot_visits']

    return df.groupby('local_date').apply(
        lambda x: (x.tot_visits - p*x.tot_visits.sum()/m)/(1-p)
        ).reset_index(drop=True)


def re_estimate_avg_duration_by_day(df, eps, categories=None):
    
    exp_eps = np.exp(eps/2)  # Divide budget by 2 (visit category + duration)

    if categories is None:
        categories = df["top_category"].unique()
    m = len(categories)
    p = 1/(1 + (exp_eps-1)/m)

    df['tot_visits'] = df.groupby('local_date').apply(
        lambda x: (x.tot_visits - p*x.tot_visits.sum()/m)
        ).reset_index(drop=True)
    
    df['tot_min_dwell'] = df.groupby('local_date').apply(
        lambda x: (x.tot_min_dwell - p*x.tot_min_dwell.sum()/m)
        ).reset_index(drop=True)
    

    df['pred_est'] = df['tot_min_dwell']/df['tot_visits']

    return df.drop(columns={'eps', 'tot_visits', 'tot_min_dwell'})


def re_estimate_tot_visits_histogram(dft, eps):
    top_categories = dft["top_category"].unique()
    exp_eps = np.exp(eps/2)
    p = 1/(1 + (exp_eps-1)/len(top_categories))

    avg_visits_overall = dft["tot_visits"].mean()
    tot_visits_est = (dft['tot_visits'] - p * avg_visits_overall)/(1-p)

    return tot_visits_est

def re_estimate_avg_duration_histogram(dft, eps):
    top_categories = dft["top_category"].unique()
    exp_eps = np.exp(eps/2)
    p = 1/(1 + (exp_eps-1)/len(top_categories))

    avg_visits_overall = dft["tot_visits"].mean()
    avg_dwell_averall = dft['tot_min_dwell'].mean()
    tot_visits_est = dft['tot_visits'] - p * avg_visits_overall#/(1-p)
    tot_dwell_est = dft['tot_min_dwell'] - p * avg_dwell_averall

    return tot_dwell_est / tot_visits_est


def re_estimate_tot_visits_by_column(dft, eps, categories=None, aggr_by='state'):
    
    exp_eps = np.exp(eps/2)  # Divide budget by 2 (category + dwell)
    if categories is None:
        categories = dft["top_category"].unique()
    m = len(categories)
    p = 1/(1 + (exp_eps-1)/m)

    dft['tot_visits'] = dft.groupby(aggr_by).apply(
        lambda x: (x.tot_visits - p*x.tot_visits.sum()/m)/(1-p),
        include_groups=False
        ).reset_index(drop=True)    
    
    dft['pred_est'] = dft['tot_visits']

    return dft


def re_estimate_avg_duration_by_column(dft, eps, categories=None, aggr_by='state'):
    
    exp_eps = np.exp(eps/2)  # Divide budget by 2 (category + dwell)
    if categories is None:
        categories = dft["top_category"].unique()
    m = len(categories)
    p = 1/(1 + (exp_eps-1)/m)

    dft['tot_visits'] = dft.groupby(aggr_by).apply(
        lambda x: (x.tot_visits - p*x.tot_visits.sum()/m)/(1-p), 
        include_groups=False
        ).reset_index(drop=True)
    
    dft['tot_min_dwell'] = dft.groupby(aggr_by).apply(
        lambda x: (x.tot_min_dwell - p*x.tot_min_dwell.sum()/m)/(1-p), 
        include_groups=False
        ).reset_index(drop=True)
    
    dft['pred_est'] = dft['tot_min_dwell']/dft['tot_visits']

    return dft


def test_function(filename: str):
    """
    Function to test this module on a CSV file (must include 'id', 'category', and 'min_dwell' columns).
    
    Args:
        filename (str): Path of the CSV file.
    """

    from utils import load_dataframe

    df = load_dataframe(filename)

    df_counts = pd.read_csv("top_categories_counts.csv")
    df_counts = df_counts.melt().sort_values("value", ascending=False).reset_index(drop=True)
    top_categories = df_counts.loc[:29]["variable"].values.tolist()

    df_priv = randomize_dataset(df, 16, top_categories)

    print(df.head())
    print(df_priv.head())


if __name__ == "__main__":

    

    keep_columns = {
        'caid': 'object',
        'top_category': 'object',
        'state': 'category',
        'local_timestamp': 'int64',
        'location_name': 'object',
        'minimum_dwell': 'int16'
    }


    filename = 'data/safegraph/visits/2020/apr/03/part-00000-tid-7859273448646466454-6fbde714-b656-46c6-bc8d-d98eebb3aba8-4053-c000.csv'
    
    
