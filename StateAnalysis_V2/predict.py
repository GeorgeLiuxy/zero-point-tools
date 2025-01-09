import os

import joblib  # 导入joblib用于保存和加载scaler
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# 目录设置
predication_data_dir = "./prediction_data/"
scaler_path = './saved_model/scaler.pkl'
model_save_path = "./saved_model/satellite_model_v2.keras"
scaler_save_path = "./saved_model/scaler.pkl"  # 保存scaler的路径


def preprocess_data(file_path, sequence_length=10, scaler=None, fit_scaler=True, scaler_path=None):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 打印列名，帮助调试
    print(f"Columns in the file: {df.columns}")

    # 检查是否包含 'Category' 列
    if 'Category' in df.columns:
        categories = df['Category'].fillna(0).values  # 将空值填充为0
    else:
        print("'Category' column not found. Setting default category value to 0.")
        categories = np.zeros(len(df))  # 如果没有 'Category' 列，创建一个全是0的数组

    # 检查是否包含 'Value' 列
    if 'Value' not in df.columns:
        raise ValueError("'Value' column not found in the data.")

    # 检查是否包含 'Timestamp' 列为
    if 'Timestamp' not in df.columns:
        raise ValueError("'Timestamp' column not found in the data.")

    # 转换 'Timestamp' 列为 datetime 类型
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # 从时间戳中提取一些有用的特征
    df['hour'] = df['Timestamp'].dt.hour  # 提取小时
    df['day'] = df['Timestamp'].dt.day  # 提取天
    df['month'] = df['Timestamp'].dt.month  # 提取月
    df['weekday'] = df['Timestamp'].dt.weekday  # 提取星期几 (0=Monday, 6=Sunday)

    # 现在将这些时间特征与 'Value' 特征结合起来
    time_features = df[['hour', 'day', 'month', 'weekday']].values
    values = df['Value'].values

    # 如果没有传入 scaler 且给定了 scaler 路径
    if scaler is None and scaler_path is not None:
        # 判断文件是否存在
        if os.path.exists(scaler_path):
            # 加载已保存的scaler
            scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
        else:
            print(f"Scaler file {scaler_path} does not exist. A new scaler will be created.")
            scaler = MinMaxScaler()  # 如果没有找到文件，则重新创建一个新的scaler

    # 如果需要拟合scaler，或者加载的是一个新的scaler
    if fit_scaler:
        # 如果是新的scaler，进行拟合
        if scaler is None:
            scaler = MinMaxScaler()
        values_scaled = scaler.fit_transform(values.reshape(-1, 1))
        # 保存新的scaler
        if scaler_path:
            joblib.dump(scaler, scaler_path)
            print(f"Scaler saved to {scaler_path}")
    else:
        # 如果不需要拟合，直接进行转换
        values_scaled = scaler.transform(values.reshape(-1, 1))

    # 使用滑动窗口来生成 X 和 y
    X, y = [], []
    for i in range(len(values_scaled) - sequence_length):
        # 将时间特征和数值特征组合成一个输入特征
        sequence_values = np.hstack([values_scaled[i:i + sequence_length], time_features[i:i + sequence_length]])  # 合并时间特征和数值特征
        X.append(sequence_values)  # 每个序列的特征
        y.append(categories[i + sequence_length])  # 对应的标签

    y = np.nan_to_num(y, nan=0)  # 处理NaN值
    y = tf.keras.utils.to_categorical(y, num_classes=4)  # 将标签进行独热编码

    # 将 X 和 y 转换为 numpy 数组，X 的形状是 (num_samples, sequence_length, num_features)
    X = np.array(X)
    y = np.array(y)

    return X, y, scaler


# 预测新数据并回填预测结果到CSV文件
def predict_new_data(file_path, output_file_path, scaler_path):
    # 加载训练好的模型
    model = load_model(model_save_path)
    print(f"Model loaded from {model_save_path}!")
    # 加载并预处理新数据
    X_new, _, scaler = preprocess_data(file_path, fit_scaler=False, scaler_path=scaler_path)
    # 对新数据进行预测
    y_pred = np.argmax(model.predict(X_new), axis=1)
    # 读取原始数据
    df_new = pd.read_csv(file_path)
    print(f"Original data length: {len(df_new)}")
    # 创建一个与原始数据行数一致的 NaN 数组
    y_pred_full = np.full(len(df_new), np.nan)
    # 填充预测结果
    y_pred_full[len(y_pred_full) - len(y_pred):] = y_pred
    df_new['category'] = y_pred_full

    # 保存带有预测结果的新文件
    df_new.to_csv(output_file_path, index=False)
    print(f"Prediction results saved to {output_file_path}")


# 选择是否进行训练或者评估
if __name__ == "__main__":

    # 预测新数据：如果要进行预测，取消下面的注释并提供文件路径
    predict_new_data(
        f'{predication_data_dir}28912.csv',  # file_path
        f'{predication_data_dir}28912_with_predictions.csv',  # output_file_path
        scaler_path
    )

