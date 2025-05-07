import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# 设置起始时间和时间间隔（每分钟一个数据点）
start_time = datetime(2023, 1, 1)  # 设置起始时间
end_time = datetime(2025, 1, 1)    # 设置结束时间
time_interval = timedelta(minutes=1)  # 每分钟生成一个数据点

# 生成时间序列
time_series = []
current_time = start_time
while current_time <= end_time:
    time_series.append(current_time)
    current_time += time_interval

# 模拟轨道半长轴
np.random.seed(42)

# 初始半长轴值（单位：km），假设正常轨道的半长轴初始值在7000-7200km之间
base_half_axis = 7100  # 初始半长轴，单位km
normal_variation_range = 2  # 正常范围内的波动幅度（单位：km）

# 初始化半长轴的变化数据
half_axis_values = []

# 模拟轨道变化
for t in range(len(time_series)):
    # 在稳定轨道期间，小幅度波动
    if t < 1000:  # 前1000分钟为稳定轨道
        variation = np.random.normal(0, normal_variation_range)  # 稳定轨道的波动
    elif 1000 <= t < 2000:  # 第1000到2000分钟为变轨阶段
        variation = np.random.normal(5, 3)  # 变轨阶段，增加较大的扰动
    else:  # 变轨完成后，回到稳定轨道
        variation = np.random.normal(0, normal_variation_range)  # 稳定轨道的波动

    half_axis_values.append(base_half_axis + variation)

# 模拟状态分类（0：没有变轨，1：变轨中）
categories = np.zeros(len(time_series))  # 初始状态：没有变轨
categories[1000:2000] = 1  # 在1000-2000分钟期间卫星变轨

# 将数据存入 DataFrame
data = pd.DataFrame({
    'Timestamp': time_series,
    'Half_Axis': half_axis_values,
    'Category': categories
})

# 显示前几行数据
print(data.head())

# 保存数据为CSV文件
data.to_csv("satellite_simulation_data_v2.csv", index=False)

# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(data['Timestamp'], data['Half_Axis'], label='Half Axis (km)', color='blue')
plt.axvspan(data['Timestamp'][1000], data['Timestamp'][2000], color='red', alpha=0.3, label='Orbit Change')
plt.title('Satellite Half Axis over Time with Orbit Change Period')
plt.xlabel('Time')
plt.ylabel('Half Axis (km)')
plt.legend()
plt.grid(True)
plt.show()
