import pandas as pd
import os

import joblib  # 用于保存和加载缩放器
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Conv1D
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# 目录设置
train_data_dir = "./train_data/"
eval_data_dir = "./evaluation_data/"
new_data_dir = "./prediction_data/"
model_save_path = "./saved_model/satellite_model_v2.keras"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)


def preprocess_data(file_path, sequence_length=10, scaler=None, fit_scaler=True):
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 提取 timestamp, value, category
    timestamps = pd.to_datetime(df['Timestamp'])
    values = df['Value'].values
    categories = df['Category'].values

    # 处理时间戳字段，提取日期时间特征（例如小时、分钟、星期几等）
    df['hour'] = timestamps.dt.hour
    df['minute'] = timestamps.dt.minute
    df['day'] = timestamps.dt.day
    df['month'] = timestamps.dt.month
    df['weekday'] = timestamps.dt.weekday  # 0: Monday, 6: Sunday

    # 周期性转换：将时间特征转换为 sin/cos 特征，捕捉周期性
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

    # 将周期性转换后的时间特征与 Value 字段合并
    time_features = df[['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']].values

    # 归一化 Value 字段
    if scaler is None:
        scaler = MinMaxScaler()
    if fit_scaler:
        values_scaled = scaler.fit_transform(values.reshape(-1, 1))
    else:
        values_scaled = scaler.transform(values.reshape(-1, 1))

    # 将归一化后的 value 和时间特征合并为模型的输入特征
    features = np.concatenate([values_scaled, time_features], axis=1)

    # 创建输入序列 (X) 和目标标签 (y)
    X, y = [], []
    for i in range(len(features) - sequence_length):
        # 每个样本包含 sequence_length 个时间步的特征
        X.append(features[i:i + sequence_length])
        # 标签是 sequence_length 之后的 category
        y.append(categories[i + sequence_length])

    X = np.array(X)

    # 对 category 进行独热编码
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = tf.keras.utils.to_categorical(y, num_classes=len(np.unique(y)))  # 自动推测类别数目

    return X, y, scaler


def build_lstm_model(input_shape, num_classes):
    # 构建 LSTM 模型
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))  # softmax 用于多分类

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


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


def train():
    global model, history
    # 数据准备
    X_train, y_train, scaler = preprocess_data(f"{train_data_dir}Satellite_A_training_data.csv")
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    X_test, y_test, _ = preprocess_data(f"{eval_data_dir}Satellite_A_evaluation_data.csv", scaler=scaler,
                                        fit_scaler=False)
    # 类别权重计算
    y_train_classes = np.argmax(y_train, axis=1)
    # 计算每个类别的权重，避免除以零
    class_weights = {}
    for i in range(4):
        class_count = np.sum(y_train_classes == i)
        if class_count == 0:
            class_weights[i] = 1.0  # 设置缺失类别的权重为1.0，或选择为0或其他默认值
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
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, lr_reduction],
        class_weight=class_weights
    )
    # 保存模型
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}!")
    # 模型评估
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    # 分类报告
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    # 检查 y_true 和 y_pred 中实际的类别数
    print(f"Unique classes in y_true: {np.unique(y_true)}")
    print(f"Unique classes in y_pred: {np.unique(y_pred)}")
    # 分类报告，添加 zero_division 参数
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    # 根据实际类别数来调整 target_names
    num_classes_in_true = len(np.unique(y_true))
    target_names = ['Uncontrolled', 'Controlled maintain', 'Controlled raising', 'Controlled lowering'][
                   :num_classes_in_true]
    # 获取所有可能的类别标签
    labels = [0, 1, 2, 3]  # 假设你有四个类别，0到3
    # 计算混淆矩阵，并指定labels参数
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    # 可视化混淆矩阵
    disp = ConfusionMatrixDisplay(conf_matrix,
                                  display_labels=['Uncontrolled', 'Controlled maintain', 'Controlled raising',
                                                  'Controlled lowering'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    # 可视化训练结果
    plt.figure(figsize=(12, 8))
    # 损失函数变化
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # 准确率变化
    plt.subplot(2, 1, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # 可视化预测结果
    plt.figure(figsize=(10, 6))
    # 使用不同的颜色和透明度来避免重叠
    plt.plot(y_true, label="True Categories", color='b', alpha=0.5)  # 蓝色，透明度设置为 0.5
    plt.plot(y_pred, label="Predicted Categories", color='orange', alpha=0.5)  # 橙色，透明度设置为 0.5
    plt.legend()
    plt.title("True vs Predicted Categories")
    plt.show()


def predict():
    # 加载模型
    if not os.path.exists(model_save_path):
        print(f"Model file not found at {model_save_path}. Please train the model first.")
        return
    model = load_model(model_save_path)
    print(f"Model loaded from {model_save_path}!")

    # 读取新数据并创建序列
    X_new, _ = preprocess_data(f"{new_data_dir}Satellite_A_new_data.csv", sequence_length=10)
    print(f"X_new shape: {X_new.shape}")

    # 进行预测
    predictions = model.predict(X_new)
    y_pred = np.argmax(predictions, axis=1)

    # 解码为类别标签
    category_labels = [index_to_class[idx] for idx in y_pred]

    # 打印或保存预测结果
    df_new = pd.read_csv(f"{new_data_dir}Satellite_A_new_data.csv")
    prediction_indices = np.arange(10, len(df_new))
    df_predictions = pd.DataFrame({
        'Timestamp': df_new['Timestamp'][prediction_indices].reset_index(drop=True),
        'Predicted_Category': category_labels
    })

    # 保存预测结果到 CSV
    os.makedirs(os.path.dirname(predictions_save_path), exist_ok=True)
    df_predictions.to_csv(predictions_save_path, index=False)
    print(f"Predictions saved to {predictions_save_path}!")

    # 可视化预测结果
    plt.figure(figsize=(10, 6))
    plt.plot(df_predictions['Timestamp'], df_predictions['Predicted_Category'],
             label="Predicted Categories", color='orange')
    plt.xlabel('Timestamp')
    plt.ylabel('Category')
    plt.title('Predicted Categories Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # train()
    predict()