import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 读取 CSV 数据
df = pd.read_csv('../cnn_data/10294.csv', parse_dates=['Timestamp'], index_col='Timestamp')

# 提取数值和类别列
values = df['Value'].values
categories = df['Category'].values

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
values_scaled = scaler.fit_transform(values.reshape(-1, 1))

# 定义函数来创建时间序列数据
def create_sequences_with_class(data, categories, seq_length):
    X, y_value, y_class = [], [], []
    for i in range(len(data) - seq_length):
        # 拼接数值数据和类别数据作为特征输入
        features = np.column_stack((data[i:i + seq_length], categories[i:i + seq_length]))
        X.append(features)
        y_value.append(data[i + seq_length])
        y_class.append(categories[i + seq_length])
    return np.array(X), np.array(y_value), np.array(y_class)

# 设置序列长度
sequence_length = 10

# 创建时间序列数据
X, y_value, y_class = create_sequences_with_class(values_scaled, categories, sequence_length)

# 将类别标签转换为独热编码
# num_classes = len(np.unique(y_class))
num_classes = 4
y_class = tf.keras.utils.to_categorical(y_class, num_classes=num_classes)

# 数据集拆分
X_train, X_test, y_value_train, y_value_test, y_class_train, y_class_test = train_test_split(
    X, y_value, y_class, test_size=0.2, random_state=42
)

# 调整数据形状以适应 CNN 和 LSTM 的输入要求
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# 定义自定义注意力层
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True)
        self.u = self.add_weight(shape=(input_shape[-1],), initializer='glorot_uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        u_t = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        a_t = tf.nn.softmax(tf.tensordot(u_t, self.u, axes=1), axis=1)
        output = inputs * tf.expand_dims(a_t, -1)
        return tf.reduce_sum(output, axis=1)

# 构建多任务模型
input_layer = Input(shape=(sequence_length, 2))

# 卷积层
cnn_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
cnn_layer = Conv1D(filters=32, kernel_size=3, activation='relu')(cnn_layer)

# LSTM 层
lstm_layer = LSTM(64, return_sequences=True)(cnn_layer)
lstm_layer = LSTM(32, return_sequences=True)(lstm_layer)

# 注意力层
attention_output = AttentionLayer()(lstm_layer)

# 数值预测分支
dense_value = Dense(32, activation='relu')(attention_output)
dense_value = Dropout(0.3)(dense_value)
output_value = Dense(1, name='value_output')(dense_value)

# 分类预测分支
dense_class = Dense(32, activation='relu')(attention_output)
dense_class = Dropout(0.3)(dense_class)
output_class = Dense(num_classes, activation='softmax', name='class_output')(dense_class)

# 构建模型
model = Model(inputs=input_layer, outputs=[output_value, output_class])

# 编译模型
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={'value_output': 'mse', 'class_output': 'categorical_crossentropy'},
    loss_weights={'value_output': 0.5, 'class_output': 0.5}
)

# 训练模型
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6)

history = model.fit(
    X_train,
    {'value_output': y_value_train, 'class_output': y_class_train},
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, lr_reduction]
)

# 模型评估
test_loss = model.evaluate(X_test, {'value_output': y_value_test, 'class_output': y_class_test})

# 预测
predictions = model.predict(X_test)
value_predictions = scaler.inverse_transform(predictions[0])
class_predictions = np.argmax(predictions[1], axis=1)
actual_values = scaler.inverse_transform(y_value_test.reshape(-1, 1))
actual_classes = np.argmax(y_class_test, axis=1)

# 绘制数值和分类预测结果对比图
plt.figure(figsize=(15, 10))

# 数值预测对比
plt.subplot(2, 1, 1)
plt.plot(actual_values, label='Actual Value', marker='o', color='blue', alpha=0.6)
plt.plot(value_predictions, label='Predicted Value', marker='x', linestyle='--', color='orange')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid(True)

# 分类预测对比
plt.subplot(2, 1, 2)
plt.plot(actual_classes, label='Actual Class', marker='o', color='green', alpha=0.6)
plt.plot(class_predictions, label='Predicted Class', marker='x', linestyle='--', color='red')
plt.title('Actual vs Predicted Classes')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



# 计算评估指标

# 1. Mean Absolute Error (MAE)
mae = mean_absolute_error(actual_values, value_predictions)
print(f'Mean Absolute Error (MAE): {mae}')

# 2. Mean Squared Error (MSE)
mse = mean_squared_error(actual_values, value_predictions)
print(f'Mean Squared Error (MSE): {mse}')

# 3. Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse}')

# 4. R² Score
r2 = r2_score(actual_values, value_predictions)
print(f'R² Score: {r2}')

accuracy = np.mean(class_predictions == actual_classes)
print(f'Classification Accuracy: {accuracy:.4f}')