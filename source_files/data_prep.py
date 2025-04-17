from data_ingestion_ADW import data
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# Removing Duplicates
data.drop_duplicates(inplace=True)

# Setting global max_date
data['COLLECTION_DATE'] = pd.to_datetime(data['COLLECTION_DATE'])
global_max_date = data['COLLECTION_DATE'].max()

# Filtering in non-zero recommended_total_qty records
data = data[data["RECOMMENDED_TOTAL_QTY"] > 0]

# Splitting CATEGORY_NAME into four new columns
split_cols = data['CATEGORY_NAME'].str.split('.', expand=True)
split_cols.columns = ['CN_1', 'CN_2', 'CN_3', 'CN_4']

# Efficiently concatenate the new columns after CATEGORY_NAME
col_position = data.columns.get_loc('CATEGORY_NAME') + 1  # Position after CATEGORY_NAME
data = pd.concat([data.iloc[:, :col_position], split_cols, data.iloc[:, col_position:]], axis=1)

data.columns = data.columns.str.lower()

# Casting columns to appropriate datatypes
data = data.astype({'recommended_total_qty':'float', 'item_code':'str', 'cn_1':'str', 'cn_2':'str', 'cn_3':'str', 'cn_4':'str', 'uom':'str'})
data['cn_4'] = data['cn_4'].fillna('NA')

# CONDITION 1(ACTIVE ITEMS): Items ordered (non-zero) atleast once in past 3 months (at plant level considering global "max_date")

def active_item_filter(df, lookback):
    df_list = []
    df = df.copy(deep=True)
    lookback_dt = global_max_date - pd.DateOffset(months = lookback)
    for org in df['org'].unique():
        df_org = df[df['org']==org]
        filtered_items = df_org[(df_org['collection_date']>=lookback_dt) & (df_org['collection_date']<=global_max_date)]['item_code'].unique()
        df_org = df_org[df_org['item_code'].isin(filtered_items)]
        if df_org.shape[0]>0:
            df_list.append(df_org)
            # print(f'org: {df_org["org"].unique()} :: Items: {df_org["item_code"].nunique()} :: Max order date: {df_org["collection_date"].max()}')
        else:
            continue
    return df_list

data_list = active_item_filter(data, 3) #(dataframe, lookback in months)
f_data = pd.concat(data_list, axis=0)


## CONDITION 2(ORDER FREQUENCY THRESHOLD): Items ordered (non-zero) atleast X times in past Y months (at plant level considering global "max_date")

def order_freq_filter(df, threshold, lookback):
    df_list = []
    df = df.copy()
    lookback_dt = global_max_date - pd.DateOffset(months = lookback)

    for org in df['org'].unique():
        df_org = df[df['org']==org]
        df_org_temp = df_org[(df_org['collection_date']>=lookback_dt) & (df_org['collection_date']<=global_max_date)]
        df_org_temp_item_order_freq = df_org_temp.groupby('item_code').size().reset_index(name="order_frequency")
        filtered_items = df_org_temp_item_order_freq[df_org_temp_item_order_freq["order_frequency"] >= threshold]["item_code"].unique()
        df_org = df_org[df_org["item_code"].isin(filtered_items)]
        
        if df_org.shape[0]>0:
            df_list.append(df_org)
            # print(f'org: {df_org["org"].unique()} :: Items: {df_org["item_code"].nunique()}')
        else:
            continue
    
    return df_list

data_list = order_freq_filter(f_data, 10, 12)
f_data = pd.concat(data_list, axis=0)

# Making item dependent mapping consistent based on the latest captured value

def fix_anomalous_values(f_data, columns_to_fix):

    f_data = f_data.copy()

    affected_item_codes = set()
    
    for col in columns_to_fix:
        t = f_data.drop_duplicates(subset=['item_code', col])
        t2 = t.groupby('item_code')[col].nunique().reset_index(name=f'unique_{col}_count')
        anomalous_items = t2[t2[f'unique_{col}_count'] > 1]['item_code'].unique()
        affected_item_codes.update(anomalous_items)

    # Step 2: Get the latest values for affected item_codes
    latest_values = f_data[f_data['item_code'].isin(affected_item_codes)] \
                    .sort_values(by=['item_code', 'collection_date'], ascending=[True, False]) \
                    .drop_duplicates(subset=['item_code'])[['item_code'] + columns_to_fix]

    # Step 3: Convert latest values into mapping dictionaries
    latest_value_dicts = {col: dict(zip(latest_values['item_code'], latest_values[col])) for col in columns_to_fix}

    # Step 4: Apply fixes in the original data
    for col in columns_to_fix:
        f_data.loc[f_data['item_code'].isin(affected_item_codes), col] = f_data['item_code'].map(latest_value_dicts[col])

    return f_data

columns_to_fix = ['cn_4', 'cn_2', 'description']
f_data_cleaned = fix_anomalous_values(f_data, columns_to_fix)

df = f_data_cleaned

df.drop(columns=["category_name","cn_1","cn_3","cn_4","description","run_period","run_no"], inplace=True)

# Cleaning string type features

df['uom'] = df['uom'].str.upper()
df['uom'] = df['uom'].str.strip()
df['cn_2'] = df['cn_2'].str.upper()
df['cn_2'] = df['cn_2'].str.strip()
df['item_code'] = df['item_code'].str.upper()
df['item_code'] = df['item_code'].str.strip()

# Maintain a list of min_date at plant level to map week_numbers against week_start_date
min_date_map = df.groupby('org')['collection_date'].min().reset_index()
min_date_map.rename(columns={'collection_date': 'min_date'}, inplace=True)
min_date_map['min_date'] = pd.to_datetime(min_date_map['min_date'].dt.date)

# Complete time series sequence

def assign_week_number(f_data):
    
    min_date = f_data['collection_date'].min()
    f_data['week_number'] = ((f_data['collection_date'] - min_date).dt.days // 7) + 1
    return f_data


def roll_up_weekly(f_data):
   
    if 'week_number' not in f_data.columns:
        raise ValueError("week_number column is missing. Ensure it is computed before rolling up.")

    # Define columns to keep (ensuring we retain unique values per item)
    columns_to_keep = ['org','item_code', 'cn_2', 'uom']

    # Drop duplicates to retain only one unique row per item
    f_data_unique = f_data.drop_duplicates(subset=['item_code'])[columns_to_keep]

    # Aggregate at item-week level
    f_data_agg = f_data.groupby(['item_code', 'week_number'], as_index=False)['recommended_total_qty'].sum()

    # Merge aggregated values back with the unique item-level details
    f_data_final = pd.merge(f_data_agg, f_data_unique, on='item_code', how='left')

    return f_data_final


def prepare_weekly_forecasting_data(f_data):
    
    # Step 1: Find the min and max week numbers at plant level
    min_week = f_data['week_number'].min()
    max_week = f_data['week_number'].max() + 4  # Extend by 4 weeks for forecasting

    # Step 2: Get all unique item codes
    all_items = f_data['item_code'].unique()

    # Step 3: Create a full DataFrame with all combinations of (item_code, week_number)
    all_weeks = range(min_week, max_week + 1)
    full_index = pd.MultiIndex.from_product([all_items, all_weeks], names=['item_code', 'week_number'])
    full_df = pd.DataFrame(index=full_index).reset_index()

    # Step 4: Merge with only the required columns to avoid duplication
    f_data_subset = f_data[['item_code', 'week_number', 'recommended_total_qty']]
    f_data_filled = pd.merge(full_df, f_data_subset, on=['item_code', 'week_number'], how='left')

    # Step 5: Fill missing values with zero for demand
    f_data_filled['recommended_total_qty'] = f_data_filled['recommended_total_qty'].fillna(0)

    # Step 6: Merge metadata from the original data, ensuring unique item mappings
    metadata_cols = ['org', 'uom', 'cn_2']
    f_data_filled = pd.merge(
        f_data_filled, 
        f_data.drop_duplicates(subset=['item_code'])[['item_code'] + metadata_cols], 
        on='item_code', 
        how='left'
    )

    return f_data_filled

def plant_level_operations(df):
    df = df.copy()
    df_list = []
    
    for org in df['org'].unique():
        df_org = df[df['org']==org]
        with_week_number_df = assign_week_number(df_org)
        rolled_up_weekly_df = roll_up_weekly(with_week_number_df)
        prepare_weekly_forecasting_data_df = prepare_weekly_forecasting_data(rolled_up_weekly_df)
        df_list.append(prepare_weekly_forecasting_data_df)

    return df_list

model_data_list = plant_level_operations(df)

# Feature Engineering

def add_features(df, lags=[1, 2, 4, 8], rolling_windows=[2, 4, 8]):
    
    df = df.copy()
    
    df.sort_values(by=["item_code", "week_number"], inplace=True)
    
    grouped = df.groupby("item_code")
    
    for lag in lags:
        df[f"lag_{lag}"] = grouped["recommended_total_qty"].shift(lag).fillna(0)  

    
    for window in rolling_windows:
        df[f"rolling_avg_{window}"] = grouped["recommended_total_qty"].transform(lambda x: x.rolling(window).mean()).fillna(method='ffill').fillna(0)
        df[f"rolling_pct_change_{window}"] = grouped["recommended_total_qty"].transform(lambda x: x.pct_change(periods=window)).replace([float("inf"), float("-inf")], 1.0).fillna(0)

    df["week_sin"] = np.sin(2 * np.pi * df["week_number"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_number"] / 52)
    
    def compute_is_live(series):
        first_nonzero_index = (series != 0).idxmax()  
        return (series.index >= first_nonzero_index).astype(int)  

    df["is_live"] = grouped["recommended_total_qty"].transform(compute_is_live)

    return df

def feature_engineering(df_list):
    df_list_final = []
    
    for df in model_data_list:
        df_list_final.append(add_features(df))
        
    return df_list_final

model_data_list_engineered = feature_engineering(model_data_list)

print("data_prep successful")
