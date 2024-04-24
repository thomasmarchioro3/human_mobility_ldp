""" Handles all CSV files of a single day and produces metadata about number of users and total dwell time """ 

import os
import numpy as np
import pandas as pd
from datetime import timedelta
from utils import load_dataframe
from calendar import monthrange

from dp_utils import randomize_dataset

from multiprocessing import Pool

eps_list = [None, 1, 2, 4, 6, 8, 16, 64]

def included_date(state, day, month):
    mandate_date = dates[ dates["State"] == state]["Public face mask mandate start"]
    if mandate_date.isna().all():
        return False
    cond1 = mandate_date - timedelta(14) > pd.to_datetime(f'2020/{month}/{day}')
    cond2 = mandate_date + timedelta(14) < pd.to_datetime(f'2020/{month}/{day}')
    return not (cond1.all() or cond2.all())

def multi(filepath, curr_states, top_categories):
    df = load_dataframe(filepath)
    df = df[df["top_category"].isin(top_categories)]
    df = df[df["state"].isin(curr_states)].reset_index(drop=True)
    df["state"] = df["state"].cat.remove_unused_categories()
    df["top_category"] = df["top_category"].cat.remove_unused_categories()
    df['idx'] = df.index

    res_list = []
    for eps in eps_list:
        if eps is not None:
            dft = randomize_dataset(df, eps, top_categories)  # apply LDP
        else:
            dft = df.copy()

        grouped = dft.groupby(["state", "local_date", "top_category"]).agg( {
            'idx': lambda x: len(x),
            'minimum_dwell': lambda y: y.sum()      
        })

        res_df_t = grouped.reset_index(drop=False)
        res_df_t = res_df_t.rename({"idx": "tot_visits", "minimum_dwell": "tot_min_dwell"}, axis = 1)
        if eps is None:
            res_df_t['eps'] = np.inf
        else:
            res_df_t['eps'] = eps

        res_list.append(res_df_t)

    res_df = pd.concat(res_list).reset_index(drop=True)

    return res_df


if __name__=="__main__":

    convert_month_name = {'01': 'jan', '02': 'feb', '03': 'mar', '04': 'apr', '05': 'may', '06': 'jun',
                        '07': 'jul', '08': 'aug', '09': 'sep', '10': 'oct', '11': 'nov', '12': 'dec'}
    
    dates = pd.read_csv('data/safegraph/facemasks.csv')

    dates["Public face mask mandate start"] = dates["Public face mask mandate start"].replace({'0':None})
    dates["Public face mask mandate start"] = pd.to_datetime(dates["Public face mask mandate start"], errors='coerce')

    states = dates["State"].unique()
    states = list(set(states.tolist())- {'District of Columbia', 'Total'})

    n_categories = 30
    df_counts = pd.read_csv("top_categories_counts.csv")
    df_counts = df_counts.melt().sort_values("value", ascending=False).reset_index(drop=True)
    top_categories = df_counts.loc[:n_categories-1]["variable"].values.tolist()

    rootdir = "./data"
    savedir = "./metadata/daily"

    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    year = "2020"
    month = "05"

    # Make list for days for the month (two-digits strings)
    n_days = monthrange(2020, int(month))[1]
    days = [f"{i:02}" for i in range(1,n_days+1)]

    for day in days:

        currdir = os.path.join(rootdir, year, convert_month_name[month], day)

        # select all states
        curr_states = [state.lower() for state in states]
        
        tot_users_dict = {state: 0 for state in curr_states}
        tot_dwell_dict = {state: 0 for state in curr_states}

        filepaths = [os.path.join(currdir, ff) for ff in os.listdir(currdir) if ff.endswith('.csv')]
        if len(filepaths) == 0:
            filepaths = [os.path.join(currdir, ff) for ff in os.listdir(currdir) if ff.endswith('.parquet')]


        def multi_f(x):
            return multi(x, curr_states, top_categories)
            
        with Pool(3) as p:
            res_dfs = p.map(multi_f, filepaths)
        

        final_df = pd.concat(res_dfs)
        final_df = final_df.groupby(['local_date', 'state', 'top_category', 'eps']).sum().reset_index(drop=False)

        final_df.to_csv(os.path.join(savedir, f"{year}_{month}_{day}.csv"), index=False)
        print(f"[Processed] {year}/{month}/{day}")



    
