import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import joblib
import os

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# 数据预处理函数
def preprocess_data(file_path, sequence_length=10, scaler=None, fit_scaler=True):
    """
    预处理数据，将时间序列转换为模型输入。

    Parameters:
    - file_path: 数据文件路径
    - sequence_length: 时间序列长度
    - scaler: 传入的标准化工具（可选）
    - fit_scaler: 是否拟合标准化工具（默认：True）

    Returns:
    - X: 模型输入
    - y: 模型目标输出（one-hot编码）
    - scaler: 归一化工具
    """
    df = pd.read_csv(file_path, parse_dates=['Timestamp'])
    values = df['Value'].values
    categories = df['Category'].values

    # 初始化或使用传入的归一化工具
    if scaler is None:
        raise ValueError("Scaler is not provided. Ensure the scaler is loaded correctly.")

    if fit_scaler:
        values_scaled = scaler.fit_transform(values.reshape(-1, 1))
    else:
        values_scaled = scaler.transform(values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(values_scaled) - sequence_length):
        X.append(values_scaled[i:i + sequence_length])
        y.append(categories[i + sequence_length])
    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, num_classes=4)
    return X, y

# 目录设置
eval_data_dir = "./evaluation_data/"
scaler_save_path = "./saved_model/scaler.pkl"
model_save_path = "./saved_model/satellite_model_v2.keras"

# 加载模型
try:
    model = load_model(model_save_path)
    print(f"Model loaded from {model_save_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 加载保存的 scaler
try:
    scaler = joblib.load(scaler_save_path)
    print(f"Scaler loaded from {scaler_save_path}")
except Exception as e:
    print(f"Error loading scaler: {e}")
    exit()

# 加载和预处理测试数据
try:
    test_file_path = f"{eval_data_dir}Satellite_A_evaluation_data.csv"
    X_test, y_test = preprocess_data(test_file_path, scaler=scaler, fit_scaler=False)
    print(f"Test data loaded and preprocessed from {test_file_path}")
except Exception as e:
    print(f"Error preprocessing test data: {e}")
    exit()

# 模型评估
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy:.4f}")

# 分类报告
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Uncontrolled', 'Controlled maintain', 'Controlled raising', 'Controlled lowering']))

# 混淆矩阵
conf_matrix = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['Uncontrolled', 'Controlled maintain', 'Controlled raising', 'Controlled lowering'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# 可视化预测结果：真值 vs 预测值
plt.figure(figsize=(12, 6))
plt.plot(y_true, label="True Categories", marker='o', linestyle='dashed')
plt.plot(y_pred, label="Predicted Categories", marker='x', linestyle='dotted')
plt.legend()
plt.title("True vs Predicted Categories")
plt.xlabel("Sample Index")
plt.ylabel("Category")
plt.grid()
plt.show()

# 可视化每一类的分类比例
unique, counts = np.unique(y_pred, return_counts=True)
plt.figure(figsize=(8, 6))
plt.bar(unique, counts, tick_label=['Uncontrolled', 'Controlled maintain', 'Controlled raising', 'Controlled lowering'])
plt.title("Predicted Category Distribution")
plt.xlabel("Category")
plt.ylabel("Frequency")
plt.show()

# 可视化分类准确性
accuracy_per_class = (conf_matrix.diagonal() / conf_matrix.sum(axis=1)) * 100
plt.figure(figsize=(8, 6))
plt.bar(range(4), accuracy_per_class, tick_label=['Uncontrolled', 'Controlled maintain', 'Controlled raising', 'Controlled lowering'])
plt.title("Accuracy per Class")
plt.xlabel("Category")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.show()
