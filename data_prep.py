import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

def load_data(file_path = r"C:\Lappy\Swapnil\ByteIQ\Motherson_Group\data_filtered_cleaned.csv"):
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    return pd.read_csv(file_path)

df = load_data()

# Assign dtypes to columns
df["collection_date"] = pd.to_datetime(df["collection_date"])
df["recommended_total_qty"] = df["recommended_total_qty"].astype(float)
df["item_code"] = df["item_code"].astype(str)

# Drop duplicates if any
df.drop_duplicates(inplace=True)
df.drop(columns=["category_name","cn_1","cn_3","cn_4","description","run_period","run_no"], inplace=True)

# Cleaning string type features

df['uom'] = df['uom'].str.upper()
df['uom'] = df['uom'].str.strip()
df['cn_2'] = df['cn_2'].str.upper()
df['cn_2'] = df['cn_2'].str.strip()
df['item_code'] = df['item_code'].str.upper()
df['item_code'] = df['item_code'].str.strip()

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

# Concatenate all dataframes into a single dataframe
# output_df = pd.concat(model_data_list_engineered, axis=0)
# output_df.to_csv(r"C:\Lappy\Swapnil\ByteIQ\Motherson_Group\engineered_data.csv", index=False)
# print("Imported Successfully")
print("Script ran successfully")
