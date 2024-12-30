import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# 生成数据并保存
train_data_dir = "./train_data/"
eval_data_dir = "./evaluation_data/"


# 数据预处理函数
def preprocess_data(file_path, sequence_length=10):
    df = pd.read_csv(file_path, parse_dates=['Timestamp'])
    values = df['Value'].values  # 只保留半长轴（semi-major axis）
    categories = df['Category'].values
    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(values_scaled) - sequence_length):
        X.append(values_scaled[i:i + sequence_length])
        y.append(categories[i + sequence_length])
    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, num_classes=4)
    return X, y, scaler

# 构建模型
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

# 预处理数据
X_train, y_train, scaler = preprocess_data(f"{train_data_dir}Satellite_A_training_data.csv")
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 构建并训练模型
model = build_model(X_train.shape[1:], num_classes=4)

# 回调函数
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6)

# 训练模型
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, lr_reduction]
)

# 保存模型
model.save("./saved_model/satellite_model_v2.keras")
print("Model saved!")

# 评估模型
X_test, y_test, _ = preprocess_data(f"{eval_data_dir}Satellite_A_evaluation_data.csv")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# 分类报告
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred))

# 混淆矩阵
conf_matrix = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['Uncontrolled', 'Controlled maintain', 'Controlled raising', 'Controlled lowering'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# 可视化结果：训练过程中的损失和准确率变化
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

# 可视化分类结果：真值与预测值的对比
plt.figure(figsize=(10, 6))
plt.plot(y_true, label="True")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("True vs Predicted Categories")
plt.show()
