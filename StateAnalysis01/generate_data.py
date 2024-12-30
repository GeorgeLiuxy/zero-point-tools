import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Helper function to generate a dataset
def generate_satellite_data(satellite_id, start_time, num_records, freq='10T'):
    """
    Generates a dataset for a single satellite with realistic control categories.

    Parameters:
    - satellite_id (str): Identifier for the satellite
    - start_time (str): Start time for the data generation
    - num_records (int): Number of records to generate
    - freq (str): Frequency of the data in pandas offset alias (e.g., '10T' for 10 minutes)

    Returns:
    - pd.DataFrame: Generated dataset
    """
    # Generate a time range
    timestamps = pd.date_range(start=start_time, periods=num_records, freq=freq)

    # Generate values (orbit semi-major axis in km) with a controlled drift pattern
    values = np.linspace(7000, 7100, num_records)  # Example range for semi-major axis

    # Add random perturbations to simulate real-world deviations
    values += np.random.normal(0, 5, num_records)

    # Generate control categories based on semi-major axis trends
    categories = []
    for value in values:
        if value < 7030:
            categories.append(0)  # Uncontrolled
        elif 7030 <= value < 7060:
            categories.append(1)  # Controlled maintain
        elif 7060 <= value < 7080:
            categories.append(2)  # Controlled raising
        else:
            categories.append(3)  # Controlled lowering

    # Create DataFrame
    data = pd.DataFrame({
        'Satellite_ID': satellite_id,
        'Timestamp': timestamps,
        'Value': values,
        'Category': categories
    })

    return data

# Directories for saving files
train_dir = "./train_data"
eval_dir = "./evaluation_data"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

# Generate multiple datasets
datasets = {}
satellites = ['Satellite_A', 'Satellite_B', 'Satellite_C', 'Satellite_D']

# Generate training datasets
for satellite in satellites:
    datasets[satellite] = generate_satellite_data(
        satellite_id=satellite,
        start_time="2023-01-01 00:00:00",
        num_records=500,  # Number of records per satellite
        freq='10T'
    )

# Save training datasets to CSV files
for satellite, data in datasets.items():
    file_path = f"{train_dir}/{satellite}_training_data.csv"
    data.to_csv(file_path, index=False)

# Generate and save evaluation datasets
evaluation_datasets = {}
for satellite in satellites:
    evaluation_datasets[satellite] = generate_satellite_data(
        satellite_id=satellite,
        start_time="2023-02-01 00:00:00",
        num_records=200,  # Number of records per satellite
        freq='10T'
    )

    file_path = f"{eval_dir}/{satellite}_evaluation_data.csv"
    evaluation_datasets[satellite].to_csv(file_path, index=False)

# Output confirmation of file paths
training_files = [f"{train_dir}/{satellite}_training_data.csv" for satellite in satellites]
evaluation_files = [f"{eval_dir}/{satellite}_evaluation_data.csv" for satellite in satellites]

print("Training Files:", training_files)
print("Evaluation Files:", evaluation_files)
