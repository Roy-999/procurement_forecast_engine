from data_prep import model_data_list_engineered
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import copy
import joblib
from pathlib import Path

# Save a copy of imported metadata
df_list = copy.deepcopy(model_data_list_engineered)

def train_xgboost(X_train, y_train, scaler):

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=6,
        random_state=43,
        enable_categorical=True
    )

    # Train the model
    model.fit(X_train, y_train)
    
    return model, scaler


def preprocess_data(df, test_weeks=4, train_window=None):

    df = df.copy(deep=True)
    
    target = "recommended_total_qty"
    features = df.columns.difference(["recommended_total_qty","org"]).tolist()


    def encode_categorical(df):
        categorical_columns = ['uom', 'item_code', 'cn_2', 'is_live']
        df[categorical_columns] = df[categorical_columns].astype('category')
        return df


    # Encode categorical variables
    df = encode_categorical(df)

    # Get the latest week number
    max_week = df["week_number"].max()
    test_start_week = max_week - test_weeks + 1

    # Apply train window restriction
    if train_window:
        train_start_week = max(test_start_week - train_window, df["week_number"].min())  
        df_train = df[(df["week_number"] >= train_start_week) & (df["week_number"] < test_start_week)]
    else:
        df_train = df[df["week_number"] < test_start_week]  # Use all weeks before the test period

    numerical_cols = [
        "week_number", "lag_1", "lag_2", "lag_4", "lag_8", 
        "rolling_avg_2", "rolling_pct_change_2","rolling_avg_4", "rolling_pct_change_4", "rolling_avg_8", 
        "rolling_pct_change_8", "week_sin", "week_cos"
    ]
    
    
    scaler = StandardScaler()
    df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])

    X_train, y_train = df_train[features], df_train[target]

    return train_xgboost(X_train, y_train, scaler)

def main_call(df_list):

    model_objects = []
    scaler_objects = []
    
    for df in df_list:
        model_obj, scaler_obj = preprocess_data(df)
        model_objects.append(model_obj)
        scaler_objects.append(scaler_obj)

    return model_objects, scaler_objects

model_objects_list, scaler_objects_list = main_call(df_list)

# Save models and scalers using joblib
current_dir = Path(__file__).resolve().parent
trained_model_path = current_dir / ".." / "source_files" / "trained_model" / "models_and_scalers.pkl"
trained_model_path = trained_model_path.resolve()
joblib.dump((model_objects_list, scaler_objects_list), trained_model_path)

print("Training successful")

dummy_variable = False