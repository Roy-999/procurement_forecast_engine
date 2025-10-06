import oci
import pandas as pd
import io
import concurrent.futures
from pathlib import Path

# Set up OCI config, the values are generated upon generating API key
current_dir = Path(__file__).resolve().parent
secret_file_path = current_dir / ".." / "source_files" / "swapnil.roy@infolob.com_2025-02-28T11_09_10.611Z.pem"
secret_file_path = secret_file_path.resolve()
config = {
    "user": user, 
    "key_file": secret_file_path,
    "tenancy": tenacy, 
    "fingerprint": fingerprint,
    "region": region
}

# Create Object Storage client
object_storage = oci.object_storage.ObjectStorageClient(config)

# Bucket and namespace details
namespace = namespace 
bucket_name = bucket_name
folder_prefix = folder_prefix  # Folder inside the bucket

# Get list of CSV files in the bucket
file_list = [
    obj.name for obj in object_storage.list_objects(namespace, bucket_name, prefix=folder_prefix).data.objects
    if obj.name.endswith(".csv")
]


# Function to fetch and read CSV as a DataFrame
def get_csv_as_dataframe(object_name):
    try:
        obj = object_storage.get_object(namespace, bucket_name, object_name)
        return pd.read_csv(io.BytesIO(obj.data.content))
    except Exception as e:
        print(f"Error loading {object_name}: {e}")
        return None

# Read CSVs in parallel using ThreadPoolExecutor
df_list = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(get_csv_as_dataframe, file_list))

# Remove None values (failed loads)
df_list = [df for df in results if df is not None]

# Combine all DataFrames
if df_list:
    data = pd.concat(df_list, ignore_index=True)

else:
    print("No valid data")

print("data_ingestion successful")

