import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# 生成轨道数据函数
def generate_orbit_data(start_date, num_days=30):
    # 初始半长轴值（单位：千米）
    initial_semi_major_axis = 7000.0
    data = []

    for i in range(num_days):
        date = start_date + timedelta(days=i)

        # 控制状态（受控保持、升轨、降轨）根据随机变化设定
        if i < 10:  # 假设前10天为受控保持
            semi_major_axis = initial_semi_major_axis + np.random.uniform(-0.5, 0.5)
            control_state = 0  # 受控保持
        elif i < 20:  # 假设第11-20天为升轨
            semi_major_axis = initial_semi_major_axis + np.random.uniform(0.5, 2.0)
            control_state = 1  # 升轨
        else:  # 假设第21-30天为降轨
            semi_major_axis = initial_semi_major_axis + np.random.uniform(-2.0, -0.5)
            control_state = 2  # 降轨

        data.append([date, semi_major_axis, control_state])
        initial_semi_major_axis = semi_major_axis  # 更新半长轴值

    df = pd.DataFrame(data, columns=['timestamp', 'semi_major_axis', 'control_state'])
    return df


# 计算变轨判定阈值（基于30天内轨道变化的中位数）
def calculate_threshold(df):
    day_changes = []
    for i in range(1, len(df)):
        day1 = df.iloc[i-1]
        day2 = df.iloc[i]

        # 计算当天的半长轴变化量
        change = abs(day2['semi_major_axis'] - day1['semi_major_axis'])
        day_changes.append(change)

    # 取过去30天变化量的中位数作为阈值
    threshold = np.median(day_changes)
    return threshold


# 变轨行为分析：标记变轨状态
def analyze_behavior(df, threshold):
    df['flag'] = 0  # 标志位，初始为0
    df['status'] = 0  # 状态标识，初始为0（受控保持）

    for i in range(1, len(df)-1):
        prev_day = df.iloc[i-1]
        current_day = df.iloc[i]
        next_day = df.iloc[i+1]

        # 计算前后两天的变化量
        diff_prev = abs(current_day['semi_major_axis'] - prev_day['semi_major_axis'])
        diff_next = abs(next_day['semi_major_axis'] - current_day['semi_major_axis'])

        # 变轨判定：当差值大于等于阈值时，认为发生变轨
        if diff_prev >= 2 * threshold or diff_next >= 2 * threshold:
            df.at[i, 'flag'] = 1  # 变轨行为发生

            if next_day['semi_major_axis'] > current_day['semi_major_axis']:
                df.at[i, 'status'] = 1  # 升轨
            else:
                df.at[i, 'status'] = 2  # 降轨
        else:
            df.at[i, 'status'] = 0  # 受控保持

    return df


# 计算转折点
def calculate_turning_point(df, i):
    """基于懒人判定或勤快人判定，计算转折点"""
    prev_day = df.iloc[i-1]
    current_day = df.iloc[i]
    next_day = df.iloc[i+1]

    # 计算斜率
    slope_prev = (current_day['semi_major_axis'] - prev_day['semi_major_axis'])
    slope_next = (next_day['semi_major_axis'] - current_day['semi_major_axis'])

    if abs(slope_prev) > abs(slope_next):
        # 取前一日作为转折点
        return i - 1
    else:
        # 取下一日作为转折点
        return i + 1


# 处理整个数据分析过程
def process_orbit_data(start_date, num_days=30):
    # 生成轨道数据
    df = generate_orbit_data(start_date, num_days)

    # 计算判定阈值
    threshold = calculate_threshold(df)

    # 分析变轨行为
    df = analyze_behavior(df, threshold)

    # 根据状态标识更新转折点（懒人判定方式）
    for i in range(1, len(df)-1):
        if df.at[i, 'flag'] == 1:
            turning_point = calculate_turning_point(df, i)
            df.at[turning_point, 'status'] = df.at[i, 'status']  # 标记转折点后的数据

    return df


# 主程序入口
if __name__ == "__main__":
    start_date = datetime(2024, 1, 1)  # 起始日期
    num_days = 30  # 数据的天数（30天）

    result_df = process_orbit_data(start_date, num_days)
    print(result_df)
