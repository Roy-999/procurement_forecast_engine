from data_prep import model_data_list_engineered
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import copy
import joblib

# Save a copy of imported metadata
df_list = copy.deepcopy(model_data_list_engineered)

# Importing pickle file with model and scaler objects
model_objects, scaler_objects = joblib.load(r"C:\Lappy\Swapnil\ByteIQ\Motherson_Group\models_and_scalers.pkl")

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

    return df

def forecast(df_list, model_objects, scaler_objects):
    forecasted_df_list = []
    for i in range(len(df_list)):
        forecasted_df_list.append(recursive_forecast(df_list[i], model_objects[i], scaler_objects[i]))
    return forecasted_df_list

forecasted_df_list = forecast(df_list, model_objects, scaler_objects)


forecasted_df_list_output = pd.concat(forecasted_df_list, axis=0)

forecasted_df_list_output.to_csv(r"C:\Lappy\Swapnil\ByteIQ\Motherson_Group\forecasted_data.csv", index=False)
print("Finished_Export")
