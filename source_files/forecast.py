# Ask if model need to be retrained before forecasting or not
# train = int(input("Press 1 to retrain the model before generating forecast, else 0: "))

import sys
train = int(sys.argv[1]) if len(sys.argv) > 1 else 0

if train==1:
    print("Running training sequence")
    from training import dummy_variable
else:
    print("Running only forecasting sequence")

from datetime import datetime, timedelta
from data_prep import model_data_list_engineered, min_date_map
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import copy
import joblib
from pathlib import Path
import os
import oracledb
from sqlalchemy import create_engine, text
from sqlalchemy.types import String

# Save a copy of imported metadata
df_list = copy.deepcopy(model_data_list_engineered)

# Importing pickle file with model and scaler objects

current_dir = Path(__file__).resolve().parent
model_path = current_dir / ".." / "source_files" / "trained_model" / "models_and_scalers.pkl"
model_path = model_path.resolve()

# Load models and scalers
model_objects, scaler_objects = joblib.load(model_path)

def add_features_forecast(df, lags=[1, 2, 4, 8], rolling_windows=[2, 4, 8]):
    df = df.copy(deep=True)    
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


def recursive_forecast(df, model, scaler, lead_time=4):

    df = df.copy(deep=True)

    # Sort to maintain order
    df.sort_values(by=["item_code", "week_number"], inplace=True)

    # Identify the first forecast week (X)
    max_week = df["week_number"].max()
    forecast_start_week = max_week-4+1
    forecast_weeks = range(forecast_start_week, forecast_start_week + lead_time)

    for week in forecast_weeks:

        df = add_features_forecast(df)

        # Select the current week’s data for prediction
        forecast_current_week = df[df["week_number"] == week].copy(deep=True)

        def encode_categorical(df):
            categorical_columns = ['uom', 'item_code', 'cn_2', 'is_live']
            df[categorical_columns] = df[categorical_columns].astype('category')
            return df

        # Encode categorical variables
        forecast_current_week = encode_categorical(forecast_current_week)

        # Normalize numerical features using the trained scaler
        numerical_cols = [
        "week_number", "lag_1", "lag_2", "lag_4", "lag_8", 
        "rolling_avg_2", "rolling_pct_change_2","rolling_avg_4", "rolling_pct_change_4", "rolling_avg_8", 
        "rolling_pct_change_8", "week_sin", "week_cos"
        ]
        
        forecast_current_week[numerical_cols] = scaler.transform(forecast_current_week[numerical_cols])

        # Extract feature columns (same as during training)
        features = forecast_current_week.columns.difference(["recommended_total_qty", "org"]).tolist()
        X_forecast = forecast_current_week[features]

        # Predict demand for the week
        predictions = model.predict(X_forecast)

        # Negative predictions will be updated to zero
        predictions = np.where(predictions < 0, 0, predictions)

        # Update target variable in the original dataset for the next iteration
        df.loc[df["week_number"] == week, "recommended_total_qty"] = predictions

    df = df[df["week_number"].isin(forecast_weeks)]

    return df

def forecast(df_list, model_objects, scaler_objects):
    forecasted_df_list = []
    for i in range(len(df_list)):
        forecasted_df_list.append(recursive_forecast(df_list[i], model_objects[i], scaler_objects[i]))
    return forecasted_df_list

forecasted_df_list = forecast(df_list, model_objects, scaler_objects)


forecasted_df_list_output = pd.concat(forecasted_df_list, axis=0)
forecasted_df_list_output["run_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Calculate week_start_date based on week_number

def calculate_week_start_date(df):
    temp = df[["org","week_number"]].drop_duplicates()
    merge_min_date = pd.merge(temp, min_date_map, on=["org"], how="left")
    merge_min_date["start_week_date"] = (merge_min_date["min_date"] + pd.to_timedelta((merge_min_date["week_number"] - 1) * 7, unit="D")).dt.date
    df = pd.merge(df, merge_min_date[["org","week_number","start_week_date"]], on=["org","week_number"], how="left")
    return df

forecasted_df_list_output = calculate_week_start_date(forecasted_df_list_output)

forecasted_df_list_output = forecasted_df_list_output[["run_date", "org", "item_code", "week_number", "start_week_date", "recommended_total_qty"]]

# Include the run timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name = fr"forecast_{timestamp}.csv"
forecast_file_path = current_dir / ".." / "forecasted_data" / file_name
forecast_file_path = forecast_file_path.resolve()
forecasted_df_list_output.to_csv(forecast_file_path, index=False)

print("Forecast Generated")


# PUSH TO ADW

current_dir = Path(__file__).resolve().parent

# Setup Oracle Client path
os.environ["PATH"] = fr"{current_dir}\instantclient-basic-windows.x64-23.7.0.25.01\instantclient_23_7" + ";" + os.environ["PATH"]
oracledb.init_oracle_client(lib_dir=fr"{current_dir}\instantclient-basic-windows.x64-23.7.0.25.01\instantclient_23_7")

# SQLAlchemy connection string
username = "mtsl_ppe_dev"
password = "Motherson12345"
dsn = "ppepocadw_high"
wallet_password = "Motherson@12345"

connection_string = f'oracle+oracledb://{username}:{password}@{dsn}?wallet_password={wallet_password}'
engine = create_engine(connection_string)

# Function to force all columns to String
def all_string_dtypes(df):
    return {col: String(255) for col in df.columns}

# Table list and corresponding dataframes
table_data_map = {
    "FORECAST_TABLE": forecasted_df_list_output
}

drop = False  # Set to False if you want to append instead of dropping existing tables

with engine.begin() as conn:
    for table_name, df in table_data_map.items():
        result = conn.execute(
            text(f"SELECT COUNT(*) FROM user_tables WHERE table_name = UPPER('{table_name}')")
        )
        exists = result.scalar() > 0

        if exists:
            if drop:
                print(f"Table {table_name} exists, dropping...")
                conn.execute(text(f'DROP TABLE "{table_name}" CASCADE CONSTRAINTS PURGE'))
                mode = "replace"
            else:
                print(f"Table {table_name} exists, appending...")
                mode = "append"
        else:
            print(f"Creating table {table_name}...")
            mode = "replace"

        dtype_map = all_string_dtypes(df)
        df.to_sql(table_name, con=conn, if_exists=mode, index=False, dtype=dtype_map)
        # print(f"Loaded {len(df)} rows into {table_name}")

print("Forecast pushed to ADW successfully on FORECAST_TABLE")

