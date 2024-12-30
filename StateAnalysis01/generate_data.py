import os
import numpy as np
import pandas as pd

# Helper function to generate realistic satellite data
def generate_satellite_data(satellite_id, start_time, num_records, freq='10T', drift_rate=0.01, periodic_effect=True):
    """
    Generates a dataset for a single satellite with realistic control categories.

    Parameters:
    - satellite_id (str): Identifier for the satellite
    - start_time (str): Start time for the data generation
    - num_records (int): Number of records to generate
    - freq (str): Frequency of the data in pandas offset alias (e.g., '10T' for 10 minutes)
    - drift_rate (float): Rate of gradual drift for the semi-major axis
    - periodic_effect (bool): Whether to include periodic orbital effects

    Returns:
    - pd.DataFrame: Generated dataset
    """
    # Generate a time range
    timestamps = pd.date_range(start=start_time, periods=num_records, freq=freq)

    # Initialize base values for semi-major axis (in km)
    base_value = 7000  # Example initial semi-major axis in kilometers

    # Gradual drift component (linear)
    drift = np.linspace(0, drift_rate * num_records, num_records)

    # Periodic variation (simulating effects such as Earth oblateness or solar radiation)
    periodic_variation = (
        np.sin(np.linspace(0, 2 * np.pi * (num_records // 144), num_records)) * 15 if periodic_effect else 0
    )

    # Random noise to simulate environmental and measurement noise
    noise = np.random.normal(0, 10, num_records)

    # Calculate the semi-major axis values
    values = base_value + drift + periodic_variation + noise

    # Generate control categories based on semi-major axis trends
    categories = []
    control_status = []  # To simulate transitions in control state

    for i, value in enumerate(values):
        # Define control state transitions based on value ranges and dynamics
        if value < 7030:
            categories.append(0)  # Uncontrolled
            control_status.append("Uncontrolled: Monitoring required")
        elif 7030 <= value < 7060:
            if i > 0 and categories[-1] != 1:
                control_status.append("Entering controlled maintenance")
            else:
                control_status.append("Controlled: Maintenance ongoing")
            categories.append(1)  # Controlled maintain
        elif 7060 <= value < 7080:
            if i > 0 and categories[-1] != 2:
                control_status.append("Initiating orbit raising")
            else:
                control_status.append("Controlled: Raising orbit")
            categories.append(2)  # Controlled raising
        else:
            if i > 0 and categories[-1] != 3:
                control_status.append("Initiating orbit lowering")
            else:
                control_status.append("Controlled: Lowering orbit")
            categories.append(3)  # Controlled lowering

    # Create DataFrame
    data = pd.DataFrame({
        'Satellite_ID': satellite_id,
        'Timestamp': timestamps,
        'Value': values,
        'Category': categories,
        'Control_Status': control_status
    })

    return data

# Directories for saving files
train_dir = "./train_data"
eval_dir = "./evaluation_data"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

# Satellites list
satellites = ['Satellite_A', 'Satellite_B', 'Satellite_C', 'Satellite_D']

# Generate and save training datasets
for satellite in satellites:
    train_data = generate_satellite_data(
        satellite_id=satellite,
        start_time="2023-01-01 00:00:00",
        num_records=500,  # Number of records per satellite
        freq='10T',
        drift_rate=0.05  # Ensure there is sufficient variation
    )

    # Save training data to CSV
    train_file_path = f"{train_dir}/{satellite}_training_data.csv"
    train_data.to_csv(train_file_path, index=False)
    print(f"Training data for {satellite} saved to {train_file_path}")

# Generate and save evaluation datasets
for satellite in satellites:
    eval_data = generate_satellite_data(
        satellite_id=satellite,
        start_time="2023-02-01 00:00:00",
        num_records=200,  # Number of records per satellite
        freq='10T',
        drift_rate=0.02,  # Slower drift for evaluation data
        periodic_effect=True
    )

    # Save evaluation data to CSV
    eval_file_path = f"{eval_dir}/{satellite}_evaluation_data.csv"
    eval_data.to_csv(eval_file_path, index=False)
    print(f"Evaluation data for {satellite} saved to {eval_file_path}")

# Output confirmation of file paths
training_files = [f"{train_dir}/{satellite}_training_data.csv" for satellite in satellites]
evaluation_files = [f"{eval_dir}/{satellite}_evaluation_data.csv" for satellite in satellites]

print("Training Files:", training_files)
print("Evaluation Files:", evaluation_files)
