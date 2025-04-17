import os
import oracledb
import pandas as pd

# Setup Oracle Client path
os.environ["PATH"] = fr"C:\Lappy\Swapnil\ByteIQ\Motherson_Group\instantclient-basic-windows.x64-23.7.0.25.01\instantclient_23_7" + ";" + os.environ["PATH"]
oracledb.init_oracle_client(lib_dir=fr"C:\Lappy\Swapnil\ByteIQ\Motherson_Group\instantclient-basic-windows.x64-23.7.0.25.01\instantclient_23_7")

# Connect to ADW
conn = oracledb.connect(
    user="mtsl_ppe_dev",
    password="Motherson12345",
    dsn="ppepocadw_high",
    wallet_password="Motherson@12345"
)

ascp_tables = [
    "ASCP_DATA_P01_STG",
    "ASCP_DATA_P02_STG",
    "ASCP_DATA_P03_STG",
    "ASCP_DATA_P04_STG",
    "ASCP_DATA_P05_STG",
    "ASCP_DATA_P06_STG",
    "ASCP_DATA_P07_STG",
    "ASCP_DATA_P08_STG",
    "ASCP_DATA_P09_STG",
    "ASCP_DATA_P10_STG",
    "ASCP_DATA_P11_STG",
    "ASCP_DATA_P12_STG",
    "ASCP_DATA_P14_STG",
    "ASCP_DATA_P15_STG",
    "ASCP_DATA_P16_STG",
    "ASCP_DATA_P17_STG",
    "ASCP_DATA_P18_STG",
    "ASCP_DATA_P19_STG",
    "ASCP_DATA_P19MAR_STG",
    "ASCP_DATA_P20_STG",
    "ASCP_DATA_P21_STG",
    "ASCP_DATA_P22_STG",
    "ASCP_DATA_P23_STG",
    "ASCP_DATA_P24_STG",
    "ASCP_DATA_P25_STG",
    "ASCP_DATA_P26_STG",
    "ASCP_DATA_P27_STG",
    "ASCP_DATA_P28_STG",
    "ASCP_DATA_P29_STG",
    "ASCP_DATA_P31_STG",
    "ASCP_DATA_P32_STG",
    "ASCP_DATA_P33_STG",
    "ASCP_DATA_P34_STG",
    "ASCP_DATA_P36_STG",
    "ASCP_DATA_P40_STG",
    "ASCP_DATA_P41_STG",
    "ASCP_DATA_P42_STG"
]

# Loop through each table except EXT_RAW_MATERIALS
df_list = []
counter = 0

for table in ascp_tables:
    try:
        print(f"Fetching data from: {table}")
        df = pd.read_sql(f'SELECT * FROM "{table}" WHERE RECOMMENDED_TOTAL_QTY > 0', con=conn)
        df_list.append(df)
        print(f"✅ Loaded: {table} (rows: {len(df)})")

    except Exception as e:
        print(f"❌ Failed to load table {table}: {e}")

conn.close()

data = pd.concat(df_list, axis=0)

print("data_ingestion_ADW successful")
