import numpy as np
import pandas as pd


states_dict = {'ak': 'alaska', 'al': 'alabama', 'ar': 'arkansas', 'az': 'arizona', 
               'ca': 'california', 'co': 'colorado', 'ct': 'connecticut', 'dc': 'district of columbia', 
               'de': 'delaware', 'fl': 'florida', 'ga': 'georgia', 'hi': 'hawaii', 'ia': 'iowa', 
               'id': 'idaho', 'il': 'illinois', 'in': 'indiana', 'ks': 'kansas', 'ky': 'kentucky', 
               'la': 'louisiana', 'ma': 'massachusetts', 'md': 'maryland', 'me': 'maine', 'mi': 'michigan', 
               'mn': 'minnesota', 'mo': 'missouri', 'ms': 'mississippi', 'mt': 'montana', 'nc': 'north carolina', 
               'nd': 'north dakota', 'ne': 'nebraska', 'nh': 'new hampshire', 'nj': 'new jersey', 
               'nm': 'new mexico', 'nv': 'nevada', 'ny': 'new york', 'oh': 'ohio', 'ok': 'oklahoma', 
               'or': 'oregon', 'pa': 'pennsylvania', 'ri': 'rhode island', 'sc': 'south carolina', 
               'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas', 'ut': 'utah', 'va': 'virginia', 
               'vt': 'vermont', 'wa': 'washington', 'wi': 'wisconsin', 'wv': 'west virginia', 'wy': 'wyomin'}


default_keep_columns = {
        'caid': 'object',
        'top_category': 'object',
        'sub_category': 'object',
        'state': 'category',
        'local_timestamp': 'int64',
        'location_name': 'object',
        'minimum_dwell': 'int16'
    }

category_names = {"Automobile Dealers": "Automobile Dealers", 
                 "Automotive Parts, Accessories, and Tire Stores": "Automotive Parts and Tire Stores", 
                 "Automotive Repair and Maintenance": "Automotive Repair", 
                 "Beer, Wine, and Liquor Stores": "Liquor Stores", 
                 "Building Material and Supplies Dealers": "Building Material and Supplies Dealers", 
                 "Clothing Stores": "Clothing Stores", 
                 "Colleges, Universities, and Professional Schools": "Colleges and Universities", 
                 "Department Stores": "Department Stores", 
                 "Depository Credit Intermediation": "Depository Credit Intermediation", 
                 "Electronics and Appliance Stores": "Electronics and Appliance Stores", 
                 "Florists": "Florists", 
                 "Furniture Stores": "Furniture Stores", 
                 "Gasoline Stations": "Gasoline Stations", 
                 "General Merchandise Stores, including Warehouse Clubs and Supercenters": "General Merchandise Stores", 
                 "Grocery Stores": "Grocery Stores", 
                 "Health and Personal Care Stores": "Health and Personal Care Stores", 
                 "Justice, Public Order, and Safety Activities": "Justice and Public Order", 
                 "Museums, Historical Sites, and Similar Institutions": "Museums and Historical Sites", 
                 "Office Supplies, Stationery, and Gift Stores": "Office Supplies and Gift Stores", 
                 "Other Amusement and Recreation Industries": "Amusement and Recreation", 
                 "Other Miscellaneous Store Retailers": "Miscellaneous Store Retailers", 
                 "Other Motor Vehicle Dealers": "Motor Vehicle Dealers", 
                 "Personal Care Services": "Personal Care Services", 
                 "Postal Service": "Postal Service", 
                 "Religious Organizations": "Religious Organizations", 
                 "Restaurants and Other Eating Places": "Restaurants and Other Eating Places", 
                 "Specialty Food Stores": "Specialty Food Stores", 
                 "Sporting Goods, Hobby, and Musical Instrument Stores": "Sporting Goods and Hobby Stores", 
                 "Traveler Accommodation": "Traveler Accommodation", 
                 "Used Merchandise Stores": "Used Merchandise Stores"}


def cast_int(x: float):
    """
    Casts to int if the number is an integer.
    """
    return int(x) if x % 1 == 0 else x


def fix_state_names(df):
    """
    Replaces the names of each state with the name in the state_dict
    """
    df = df.replace({'state': states_dict})
    df = df[df['state'].isin(states_dict.values())].reset_index(drop=True)
    return df



def get_key_event_day_range_per_state(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["eps"] == np.inf].reset_index(drop=True) # Choose 1 example epsilon, to get number of days per state

    before = df.copy()
    before = df[df["difference_from_mandate"] < 0].reset_index(drop=True)
    after = df.copy()
    after = df[df["difference_from_mandate"] > 0].reset_index(drop=True)
    df_states_days = pd.DataFrame(columns = ["state", "before_after", "total"])

    for state in df["state"].unique():
        temp_1 = before[before["state"] == state].reset_index(drop=True)
        difference_before = temp_1["difference_from_mandate"].unique()
        df_states_days = df_states_days._append({"state": state, "before_after": "before", "total": len(difference_before)}, ignore_index=True)
        temp_2 = after[after["state"] == state].reset_index(drop=True)
        difference_after = temp_2["difference_from_mandate"].unique()
        df_states_days = df_states_days._append({"state": state, "before_after": "after", "total": len(difference_after)}, ignore_index=True)
    return df_states_days

def get_states_within_range(df: pd.DataFrame, day_threshold=14) -> list:

    days_per_state = get_key_event_day_range_per_state(df)
    # discard states that do not have 14 before and after
    keep_states_before = days_per_state[
        (days_per_state["total"] == day_threshold)&(days_per_state['before_after']=='before')]["state"].to_list()
    keep_states_after = days_per_state[
        (days_per_state["total"] == day_threshold)&(days_per_state['before_after']=='after')]["state"].to_list()
    keep_states = sorted(list(set(keep_states_before).intersection(set(keep_states_after))))
    return keep_states

def load_dataframe(filename, keep_columns=default_keep_columns):
    if filename.endswith('.csv'):
        df = pd.read_csv(filename, usecols=keep_columns.keys(), dtype=keep_columns)
    elif filename.endswith('.parquet'):
        df = pd.read_parquet(filename, columns=list(keep_columns.keys()))
        df = df.astype(keep_columns)
    else:
        raise Exception('Invalid file extension.')
    df = fix_state_names(df)
    df['local_date'] = pd.to_datetime(df['local_timestamp'], unit='s').dt.date
    # df['local_date'] = pd.to_datetime(df['local_date'])
    df.loc[df['location_name'] == 'home', ['top_category','sub_category']] = 'Home' 
    df = df.astype({'top_category':'category', 'sub_category':'category'})
    df.drop(columns=['local_timestamp','location_name'], inplace=True)
    df = df.dropna()
    return df

if __name__ == "__main__":

    filename = 'data/visits/2020/apr/03/part-00000-tid-7859273448646466454-6fbde714-b656-46c6-bc8d-d98eebb3aba8-4053-c000.csv'
    
    df = load_dataframe(filename)
    print(df.info())

    # print(df['sub_category'].value_counts())
    print(df['sub_category'].unique())