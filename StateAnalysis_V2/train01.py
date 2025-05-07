import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, MultiHeadAttention, GlobalAveragePooling1D, Add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# 目录设置
train_data_dir = "./new_train_data/"
scaler_save_path = './saved_model/scaler.pkl'
model_save_path = "./saved_model/satellite_model_v4.keras"

# 创建保存目录
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)


# 构建优化后的模型
def build_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape)

    # 卷积层
    cnn_layer = Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_layer)
    cnn_layer = Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(cnn_layer)

    # LSTM层
    lstm_layer = LSTM(128, return_sequences=True)(cnn_layer)
    lstm_layer = Dropout(0.3)(lstm_layer)

    # 自注意力机制
    attention_layer = MultiHeadAttention(num_heads=4, key_dim=128)(lstm_layer, lstm_layer)
    attention_layer = Add()([lstm_layer, attention_layer])  # Add residual connection
    attention_layer = GlobalAveragePooling1D()(attention_layer)

    # Dense层
    dense_layer = Dense(64, activation='relu')(attention_layer)
    dense_layer = Dropout(0.6)(dense_layer)

    # 输出层
    output_layer = Dense(num_classes, activation='softmax')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 预处理数据函数
def preprocess_data(file_path, sequence_length=10, scaler=None, fit_scaler=True, scaler_path=None):
    df = pd.read_csv(file_path)

    # 检查是否包含 'Category' 列
    if 'Category' in df.columns:
        categories = df['Category'].fillna(0).values  # 将空值填充为0
    else:
        categories = np.zeros(len(df))  # 如果没有 'Category' 列，创建一个全是0的数组

    # 检查是否包含 'Value' 列
    if 'Value' not in df.columns:
        raise ValueError("'Value' column not found in the data.")


    # 获取状态真实值并进行归一化
    true_state = df['Category'].values
    true_state_scaled = (true_state - np.min(true_state)) / (np.max(true_state) - np.min(true_state))  # 归一化

    # 计算半长轴的变化特征
    values = df['Value'].values
    half_axis_changes = np.diff(values)  # 计算相邻帧的半长轴变化（差值）
    half_axis_changes = np.insert(half_axis_changes, 0, 0)  # 插入一个0，保持与原数据对齐

    # 加载或创建 Scaler
    if scaler is None and scaler_path is not None:
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            scaler = MinMaxScaler()

    if fit_scaler:
        values_scaled = scaler.fit_transform(values.reshape(-1, 1))
        if scaler_path:
            joblib.dump(scaler, scaler_path)
    else:
        values_scaled = scaler.transform(values.reshape(-1, 1))

    # 将半长轴的变化和状态真实值作为特征
    X, y = [], []
    for i in range(len(values_scaled) - sequence_length):
        # 将半长轴变化和状态真实值组合成序列
        sequence_values = np.hstack([values_scaled[i:i + sequence_length],
                                     half_axis_changes[i:i + sequence_length].reshape(-1, 1),  # 加入变化特征
                                     true_state_scaled[i:i + sequence_length].reshape(-1, 1)])  # 加入状态真实值
        X.append(sequence_values)
        y.append(categories[i + sequence_length])

    y = np.nan_to_num(y, nan=0)
    y = tf.keras.utils.to_categorical(y, num_classes=4)

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler


# 训练模型函数（逐文件更新）
def train_model_incrementally():
    # 获取所有 CSV 文件路径
    all_files = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir) if f.endswith('.csv')]

    # 初始化模型变量
    model = None
    scaler = None

    # 遍历每个文件进行训练
    for file_path in all_files:
        print(f"Training on file: {file_path}")

        # 处理当前文件的数据
        X_train, y_train, scaler = preprocess_data(file_path, fit_scaler=True, scaler_path=scaler_save_path)

        # 如果模型为空（即第一次训练），则构建一个新的模型
        if model is None:
            # 如果是第一次训练，构建新模型
            model = build_model(X_train.shape[1:], num_classes=4)
        else:
            # 否则加载已保存的模型
            model = load_model(model_save_path)
            print(f"Model loaded from {model_save_path}")

        # 将数据分割为训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # 训练模型
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.7, min_lr=1e-6)

        class_weights = {0: 1, 1: 2, 2: 2, 3: 2}  # 假设0类占比过高，可以增加其它类别的权重
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=16,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, lr_reduction],
            class_weight=class_weights,  # 添加类权重
            verbose=1
        )

        # 每次训练后保存更新的模型
        model.save(model_save_path)
        print(f"Model updated and saved to {model_save_path}")

        # 每次训练后保存 Scaler
        joblib.dump(scaler, scaler_save_path)
        print(f"Scaler saved to {scaler_save_path}")

        # 绘制损失和准确率曲线
        plt.figure(figsize=(12, 6))

        # 损失曲线
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


# 运行增量训练
if __name__ == "__main__":
    train_model_incrementally()
