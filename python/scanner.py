import os
import time
from datetime import datetime

# Define your dataset path
dataset_path = '/mnt/projects/dzambala_data'

# Get the current time
now = datetime.now()
folder_name = now.strftime('%Y%m%d_%H%M')  # Format: YYYYMMDD_HHMM

# Create the full path
full_path = os.path.join(dataset_path, folder_name)

# Create the folder
os.makedirs(full_path, exist_ok=True)
print(f"Created folder: {full_path}")

# Sleep for 300 seconds
time.sleep(300)
