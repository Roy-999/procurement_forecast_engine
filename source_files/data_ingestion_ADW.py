from pathlib import Path
import os
import pandas as pd
from sqlalchemy import create_engine, text
import oracledb

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

# List of ASCP tables
ascp_tables = [
    "ASCP_DATA_P01_STG", "ASCP_DATA_P02_STG", "ASCP_DATA_P03_STG", "ASCP_DATA_P04_STG",
    "ASCP_DATA_P05_STG", "ASCP_DATA_P06_STG", "ASCP_DATA_P07_STG", "ASCP_DATA_P08_STG",
    "ASCP_DATA_P09_STG", "ASCP_DATA_P10_STG", "ASCP_DATA_P11_STG", "ASCP_DATA_P12_STG",
    "ASCP_DATA_P14_STG", "ASCP_DATA_P15_STG", "ASCP_DATA_P16_STG", "ASCP_DATA_P17_STG",
    "ASCP_DATA_P18_STG", "ASCP_DATA_P19_STG", "ASCP_DATA_P19MAR_STG", "ASCP_DATA_P20_STG",
    "ASCP_DATA_P21_STG", "ASCP_DATA_P22_STG", "ASCP_DATA_P23_STG", "ASCP_DATA_P24_STG",
    "ASCP_DATA_P25_STG", "ASCP_DATA_P26_STG", "ASCP_DATA_P27_STG", "ASCP_DATA_P28_STG",
    "ASCP_DATA_P29_STG", "ASCP_DATA_P31_STG", "ASCP_DATA_P32_STG", "ASCP_DATA_P33_STG",
    "ASCP_DATA_P34_STG", "ASCP_DATA_P36_STG", "ASCP_DATA_P40_STG", "ASCP_DATA_P41_STG",
    "ASCP_DATA_P42_STG"
]

df_list = []
# Load data from each table
with engine.begin() as conn:
    for table in ascp_tables:
        try:
            print(f"Fetching data from: {table}")
            query = text(f'SELECT * FROM "{table}" WHERE RECOMMENDED_TOTAL_QTY > 0')
            df = pd.read_sql(query, conn)
            if df.empty:
                continue
            df_list.append(df)
            print(f"Loaded: {table} (rows: {len(df)})")
        except Exception as e:
            print(f"Failed to load table {table}: {e}")

# Concatenate all dataframes
data = pd.concat(df_list, axis=0)

print("data_ingestion_ADW successful")
