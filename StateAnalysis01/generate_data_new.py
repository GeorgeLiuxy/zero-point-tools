import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# 太阳辐射压力模型（简单的周期性扰动模拟）
def solar_radiation_pressure_effect(t, amplitude=0.02):
    # 模拟太阳辐射压力对轨道的周期性影响，假设影响周期为1天
    return amplitude * np.sin(2 * np.pi * t / 86400)  # 86400秒 = 1天

# J2效应模型（地球引力扰动）
def j2_effect(t, semi_major_axis, inclination=0.1, j2=1.082626e-3):
    """
    简单的J2效应模拟，假设地球的非球形引力导致轨道周期性扰动。
    """
    # 计算轨道的偏心率和倾斜度
    # 假设轨道是近圆轨道，因此偏心率较小
    eccentricity = 0.001  # 轨道偏心率
    period = np.sqrt(semi_major_axis**3 / 398600)  # 周期，单位为秒
    omega_dot = -1.5 * j2 * (398600**(1/2)) * (semi_major_axis**(-2.5))  # 轨道预cession率
    return omega_dot * np.sin(2 * np.pi * t / period)  # 产生周期性扰动

# Helper function to generate a dataset
def generate_satellite_data(start_time, num_records, freq='10T'):
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
    time_seconds = np.arange(num_records) * pd.to_timedelta(freq).total_seconds()  # Time in seconds for simulation

    # Initial semi-major axis (in km)
    semi_major_axis = 7000  # Initial semi-major axis in km
    drift_rate = 0.02  # Drift rate in km/day

    # Add random perturbations to simulate real-world deviations (e.g., from solar radiation pressure and J2 effect)
    values = np.linspace(semi_major_axis, semi_major_axis + 100, num_records)  # Drift from 7000 to 7100 km

    # Adding solar radiation pressure effect
    solar_effects = np.array([solar_radiation_pressure_effect(t) for t in time_seconds])

    # Adding J2 effect (Earth's gravity field influence)
    j2_effects = np.array([j2_effect(t, semi_major_axis) for t in time_seconds])

    # Combine the effects with random noise
    values += drift_rate * time_seconds / 86400 + solar_effects + j2_effects + np.random.normal(0, 3, num_records)

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
        'Timestamp': timestamps,
        'Value': values,
        'Category': categories
    })

    return data

def generate_satellite_prediction_data(start_time, num_records, freq='10T'):
    # Generate a time range
    timestamps = pd.date_range(start=start_time, periods=num_records, freq=freq)
    time_seconds = np.arange(num_records) * pd.to_timedelta(freq).total_seconds()  # Time in seconds for simulation

    # Initial semi-major axis (in km)
    semi_major_axis = 7000  # Initial semi-major axis in km
    drift_rate = 0.02  # Drift rate in km/day

    # Add random perturbations to simulate real-world deviations (e.g., from solar radiation pressure and J2 effect)
    values = np.linspace(semi_major_axis, semi_major_axis + 100, num_records)  # Drift from 7000 to 7100 km

    # Adding solar radiation pressure effect
    solar_effects = np.array([solar_radiation_pressure_effect(t) for t in time_seconds])

    # Adding J2 effect (Earth's gravity field influence)
    j2_effects = np.array([j2_effect(t, semi_major_axis) for t in time_seconds])

    # Combine the effects with random noise
    values += drift_rate * time_seconds / 86400 + solar_effects + j2_effects + np.random.normal(0, 3, num_records)

    # Create DataFrame
    data = pd.DataFrame({
        'Timestamp': timestamps,
        'Value': values
    })

    return data

train_dir = "./train_data"
eval_dir = "./evaluation_data"
prediction_dir = "./prediction_data"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)
os.makedirs(prediction_dir, exist_ok=True)

# Satellites list
satellites = ['Satellite_A', 'Satellite_B', 'Satellite_C', 'Satellite_D']

# Generate and save training datasets
for satellite in satellites:
    train_data = generate_satellite_data(
        start_time="2023-01-01 00:00:00",
        num_records=5000,  # Number of records per satellite
        freq='10T'
    )

    # Save training data to CSV
    train_file_path = f"{train_dir}/{satellite}_training_data.csv"
    train_data.to_csv(train_file_path, index=False)
    print(f"Training data for {satellite} saved to {train_file_path}")

# Generate and save evaluation datasets
for satellite in satellites:
    eval_data = generate_satellite_data(
        start_time="2024-02-01 00:00:00",
        num_records=200,  # Number of records per satellite
        freq='10T'
    )

    # Save evaluation data to CSV
    eval_file_path = f"{eval_dir}/{satellite}_evaluation_data.csv"
    eval_data.to_csv(eval_file_path, index=False)
    print(f"Evaluation data for {satellite} saved to {eval_file_path}")


# Generate and save prediction datasets for each satellite
for satellite in satellites:
    prediction_data = generate_satellite_prediction_data(
        start_time="2024-03-01 00:00:00",  # Start time for prediction data (can be adjusted)
        num_records=200,  # Number of records per satellite
        freq='10T'  # Time frequency
    )

    # Save prediction data to CSV (without Category)
    prediction_file_path = f"{prediction_dir}/{satellite}_prediction_data.csv"
    prediction_data.to_csv(prediction_file_path, index=False)
    print(f"Prediction data for {satellite} saved to {prediction_file_path}")

# Output confirmation of file paths
training_files = [f"{train_dir}/{satellite}_training_data.csv" for satellite in satellites]
evaluation_files = [f"{eval_dir}/{satellite}_evaluation_data.csv" for satellite in satellites]
prediction_files = [f"{prediction_dir}/{satellite}_prediction_data.csv" for satellite in satellites]

print("Training Files:", training_files)
print("Evaluation Files:", evaluation_files)
print("Prediction Files:", prediction_files)
