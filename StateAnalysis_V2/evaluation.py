import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler

# 目录设置
model_save_path = "./saved_model/satellite_model_v4.keras"
scaler_save_path = './saved_model/scaler.pkl'

# 加载模型和Scaler
model = load_model(model_save_path)
scaler = joblib.load(scaler_save_path)

# 预处理新数据
def preprocess_new_data(file_path, sequence_length=10, scaler=None):
    # 读取新数据
    df = pd.read_csv(file_path)

    # 检查是否包含 'True_State' 列
    if 'Category' not in df.columns:
        raise ValueError("'True_State' column not found in the new data.")

    # 获取状态真实值并进行归一化
    true_state = df['Category'].values
    true_state_scaled = (true_state - np.min(true_state)) / (np.max(true_state) - np.min(true_state))  # 归一化

    # 计算半长轴的变化特征
    values = df['Value'].values
    half_axis_changes = np.diff(values)  # 计算相邻帧的半长轴变化（差值）
    half_axis_changes = np.insert(half_axis_changes, 0, 0)  # 插入一个0，保持与原数据对齐

    # 使用训练时的Scaler对新数据进行缩放
    values_scaled = scaler.transform(values.reshape(-1, 1))

    # 将半长轴变化和状态真实值作为特征
    X = []
    for i in range(len(values_scaled) - sequence_length):
        # 将半长轴变化和状态真实值组合成序列
        sequence_values = np.hstack([values_scaled[i:i + sequence_length],
                                     half_axis_changes[i:i + sequence_length].reshape(-1, 1),  # 加入变化特征
                                     true_state_scaled[i:i + sequence_length].reshape(-1, 1)])  # 加入状态真实值
        X.append(sequence_values)

    X = np.array(X)
    return X

# 预测新数据
def predict_new_data(file_path):
    # 预处理新数据
    X_new = preprocess_new_data(file_path, scaler=scaler)

    # 进行预测
    predictions = model.predict(X_new)

    # 将预测结果转换为类别
    predicted_classes = np.argmax(predictions, axis=1)

    return predicted_classes, predictions

# 评估模型
def evaluate_model(true_labels, predicted_classes):
    # 打印分类报告
    print(classification_report(true_labels, predicted_classes))

    # 打印混淆矩阵
    cm = confusion_matrix(true_labels, predicted_classes)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_display.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

# 读取新数据并进行预测
if __name__ == "__main__":
    # 假设你的新数据文件路径是 "new_test_data.csv"
    new_data_file = "new_evaluation_data/16526new.csv"

    # 进行预测
    predicted_classes, predictions = predict_new_data(new_data_file)

    # 打印预测结果
    print("Predicted Classes:", predicted_classes)

    # 真实标签，假设新数据中有列 'Category' 表示真实标签
    df_new = pd.read_csv(new_data_file)
    true_labels = df_new['Category'].values[10:]  # 因为序列是从10开始的，所以真实标签从第11行开始

    # 评估模型
    evaluate_model(true_labels, predicted_classes)
