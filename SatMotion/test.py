import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# 创建时间序列数据

def create_sequences(data, categories, seq_length):
    X, y_value, y_class = [], [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y_value.append(data[i + seq_length])
        y_class.append(categories[i + seq_length])
    return np.array(X), np.array(y_value), np.array(y_class)
# 数据预处理
def preprocess_data(train_dir, value_col='Value', category_col='Category', global_num_classes=None):
    combine_training_data(train_dir)
    df = pd.read_csv(train_dir, parse_dates=['Timestamp'], index_col='Timestamp')

    # 提取数值和类别列
    values = df[value_col].values
    categories = df[category_col].values

    # 将类别值重新映射为连续的索引
    unique_categories = np.unique(categories)
    category_mapping = {cat: idx for idx, cat in enumerate(unique_categories)}
    categories = np.array([category_mapping[cat] for cat in categories])

    print(f"File: {train_dir} - Unique categories: {unique_categories}")
    print(f"Category mapping: {category_mapping}")

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    values_scaled = scaler.fit_transform(values.reshape(-1, 1))

    # 创建时间序列数据
    X, y_value, y_class = create_sequences(values_scaled, categories, sequence_length)

    # 动态计算类别数或使用全局类别数
    if global_num_classes is None:
        num_classes = len(unique_categories)
    else:
        num_classes = global_num_classes

    # 独热编码
    y_class = tf.keras.utils.to_categorical(y_class, num_classes=num_classes)

    return X, y_value, y_class, scaler

# 合并所有训练数据
def combine_training_data(directory, sequence_length, value_col='Value', category_col='Category'):
    if not os.path.isdir(directory):  # 检查是否为目录
        raise NotADirectoryError(f"Provided path '{directory}' is not a directory. Please provide a valid directory containing CSV files.")

    all_X, all_y_value, all_y_class = [], [], []
    all_categories = set()

    # 先收集所有类别
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if file_name.endswith(".csv"):
            df = pd.read_csv(file_path)
            all_categories.update(np.unique(df[category_col]))

    global_num_classes = len(all_categories)
    print(f"Total unique categories across all files: {all_categories} (num_classes={global_num_classes})")

    # 再处理每个文件
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if file_name.endswith(".csv"):
            X, y_value, y_class, _ = preprocess_data(file_path, sequence_length,
                                                     value_col=value_col,
                                                     category_col=category_col,
                                                     global_num_classes=global_num_classes)
            all_X.append(X)
            all_y_value.append(y_value)
            all_y_class.append(y_class)

    # 合并所有数据
    all_X = np.vstack(all_X)
    all_y_value = np.concatenate(all_y_value)
    all_y_class = np.vstack(all_y_class)

    return all_X, all_y_value, all_y_class

# 示例调用
train_dir = "./train_data"  # 确保这里是一个目录，而不是单个文件路径
sequence_length = 10

try:
    X, y_value, y_class = combine_training_data(train_dir, sequence_length)
    print(f"Combined training data shape: X={X.shape}, y_value={y_value.shape}, y_class={y_class.shape}")
except NotADirectoryError as e:
    print(str(e))
