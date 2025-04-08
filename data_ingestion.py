import pandas as pd
import os

def load_data(file_path = r"C:\Lappy\Swapnil\ByteIQ\Motherson_Group\Data\db.csv"):
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    return pd.read_csv(file_path)

df = load_data()
#print(df.head())
#print(df.columns)