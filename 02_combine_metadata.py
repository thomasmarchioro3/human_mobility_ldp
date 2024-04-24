import os
import pandas as pd

# calculate the difference between the local_date and the start of mask mandate
def get_difference_from_mask_mandate(row, dates):
    today = row["local_date"]
    state = row["state"]
    mandate_start = dates[dates["State"].str.lower() == state]["Public face mask mandate start"].values

    return ( pd.to_datetime(today) - pd.to_datetime(mandate_start) ).days.values[0]


if __name__ == "__main__":

    rootdir = './metadata/daily'
    files = os.listdir(rootdir)

    df = pd.concat([pd.read_csv(os.path.join(rootdir,filename)) for filename in files])
    df = df.sort_values(by = ["eps", "local_date", "state"])
    df = df.reset_index(drop=True)
    df = df.groupby(["eps", "state", "local_date", "top_category"]).sum().reset_index(drop=False)


    # discard incomplete days
    discard_dates = ["2020-03-31", "2020-04-01", "2020-05-31"]
    df = df[~df['local_date'].isin(discard_dates)].reset_index(drop=True)
    

    mask_dates = pd.read_csv('data/safegraph/facemasks.csv')
    mask_dates["Public face mask mandate start"] = mask_dates["Public face mask mandate start"].replace({'0':None})
    mask_dates["Public face mask mandate start"] = pd.to_datetime(mask_dates["Public face mask mandate start"], errors='coerce')

    mask_f = lambda x: get_difference_from_mask_mandate(x, dates=mask_dates)


    df["difference_from_mandate"] = df.apply(mask_f, axis=1)
    df = df.sort_values(by=["eps", "local_date", "difference_from_mandate", "state", "top_category"])



    # Final columns for combined metadata
    df = df[["eps", "local_date", "difference_from_mandate", "state", "top_category", "tot_visits", "tot_min_dwell"]]
    df.to_csv("metadata/combined.csv", index=False)
