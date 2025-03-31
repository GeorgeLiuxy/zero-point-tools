import os

import joblib  # 导入joblib用于保存和加载scaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# 目录设置
train_data_dir = "./new_train_data/"
eval_data_dir = "./evaluation_data/"
predication_data_dir = "./prediction_data/"
scaler_path = './saved_model/scaler.pkl'
model_save_path = "./saved_model/satellite_model_v3.keras"
scaler_save_path = "./saved_model/scaler.pkl"  # 保存scaler的路径
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# 构建模型函数
def build_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape)

    # 卷积层
    cnn_layer = Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_layer)
    cnn_layer = Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(cnn_layer)

    # LSTM层
    lstm_layer = LSTM(64, return_sequences=True)(cnn_layer)
    lstm_layer = LSTM(32)(lstm_layer)

    # Dense层
    dense_layer = Dense(32, activation='relu')(lstm_layer)
    dropout_layer = Dropout(0.3)(dense_layer)

    # 输出层
    output_layer = Dense(num_classes, activation='softmax')(dropout_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


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

    # 检查是否包含 'Timestamp' 列
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


# 训练模型函数
def train_model(file_name):
    # 数据准备
    X_train, y_train, scaler = preprocess_data(f"{file_name}", fit_scaler=True, scaler_path=scaler_save_path)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 类别权重计算
    y_train_classes = np.argmax(y_train, axis=1)
    class_weights = {}
    for i in range(4):
        class_count = np.sum(y_train_classes == i)
        if class_count == 0:
            # 如果类别的样本数为0，避免除零错误，设置权重为1.0（或其他合理值）
            class_weights[i] = 1.0
        else:
            class_weights[i] = 1.0 / class_count

    print("Class weights:", class_weights)

    # 构建模型
    model = build_model(X_train.shape[1:], num_classes=4)

    # 回调函数
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6)

    # 模型训练
    history = model.fit(
        X_train, y_train,
        epochs=1000,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, lr_reduction],
        class_weight=class_weights
    )

    # 保存模型
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}!")

    # 在训练时保存 scaler
    joblib.dump(scaler, scaler_save_path)
    print(f"Scaler saved to {scaler_save_path}!")

    # 绘制损失和准确率的变化曲线
    # 损失曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


# 评估与预测函数
def evaluate_model(file_name):
    # 加载模型
    model = load_model(model_save_path)
    print(f"Model loaded from {model_save_path}!")

    # 加载并准备数据
    X_test, y_test, scaler = preprocess_data(f"{file_name}", fit_scaler=False, scaler_path=scaler_save_path)

    # 模型评估
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # 预测结果
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    # 检查y_true中的实际类别数
    unique_classes_in_true = np.unique(y_true)
    num_classes_in_true = len(unique_classes_in_true)

    # 动态调整 target_names
    target_names = ['Uncontrolled', 'Controlled maintain', 'Controlled raising', 'Controlled lowering'][:num_classes_in_true]

    # 打印分类报告，加入 zero_division 参数
    print(classification_report(
        y_true, y_pred,
        target_names=target_names,
        labels=unique_classes_in_true,
        zero_division=0  # 可以设置为 0 或 1，取决于你希望如何处理没有预测到的类别
    ))

    # 混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    # 可视化预测结果
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="True Categories", color='b', alpha=0.5)
    plt.plot(y_pred, label="Predicted Categories", color='orange', alpha=0.5)
    plt.legend()
    plt.title("True vs Predicted Categories")
    plt.show()


# 选择是否进行训练或者评估
if __name__ == "__main__":
    # # 训练模型：如果需要训练模型，则取消下面的注释
    # train_model(f'{train_data_dir}14899new.csv')

    # 加载模型并进行评估：如果已经训练过模型且要评估，取消下面的注释
    evaluate_model(f'{train_data_dir}16526new.csv')


