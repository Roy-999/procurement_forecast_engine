import oci
import pandas as pd
import io
import concurrent.futures

# Set up OCI config, the values are generated upon generating API key
config = {
    "user": "ocid1.user.oc1..aaaaaaaabepmp6cnh7quc2kq2lp65bg5zoaaafb3i6ysliw4xtsolf5245pq", 
    "key_file": "swapnil.roy@infolob.com_2025-02-28T11_09_10.611Z.pem",
    "tenancy": "ocid1.tenancy.oc1..aaaaaaaamvz6uy5l5mdihzdqex43sp3kwdg45hzhm5djvfiqf2aq2fusbb4q", 
    "fingerprint": "54:d0:cd:22:ea:f1:2f:ca:01:62:de:58:57:0a:7d:76",
    "region": "ap-mumbai-1"
}

# Create Object Storage client
object_storage = oci.object_storage.ObjectStorageClient(config)

# Bucket and namespace details
namespace = "bmglwagxa8my"  
bucket_name = "ppepoc_landing"
folder_prefix = "ascp_cleaned_data/"  # Folder inside the bucket

# Get list of CSV files in the bucket
file_list = [
    obj.name for obj in object_storage.list_objects(namespace, bucket_name, prefix=folder_prefix).data.objects
    if obj.name.endswith(".csv")
]

print(f"Found {len(file_list)} CSV files in {folder_prefix}")

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
    print("Combined DataFrame Shape:", data.shape)

    # Basic insights
    print("Unique ORG values:", data["ORG"].nunique())
    print("Unique ITEM_CODE values:", data["ITEM_CODE"].nunique())
else:
    print("No valid CSV files found!")

print("data_ingestion ran successfully")