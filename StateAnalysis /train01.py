import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
# 读取 CSV 数据
df = pd.read_csv('train_data/10294.csv', parse_dates=['Timestamp'], index_col='Timestamp')

# 提取数值和类别列
values = df['Value'].values
categories = df['Category'].values

# 提取时间特征
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# 归一化数值列
scaler = MinMaxScaler(feature_range=(0, 1))
values_scaled = scaler.fit_transform(values.reshape(-1, 1))

# 定义滑动窗口函数，包含时间特征和数值数据
def create_sequences_with_timestamp(data, categories, timestamp_features, seq_length):
    X, y_class = [], []
    for i in range(len(data) - seq_length):
        # 提取数值序列特征
        value_features = data[i:i + seq_length]

        # 提取时间戳特征（同样是序列化的）
        time_features = timestamp_features[i:i + seq_length]

        # 合并数值和时间特征
        features = np.column_stack((value_features, time_features))

        X.append(features)
        y_class.append(categories[i + seq_length])  # 目标类别
    return np.array(X), np.array(y_class)

# 提取时间戳特征
timestamp_features = df[['hour', 'day_of_week', 'month', 'is_weekend']].values

# 设置序列长度
sequence_length = 10

# 创建时间序列数据
X, y_class = create_sequences_with_timestamp(values_scaled, categories, timestamp_features, sequence_length)

# 将类别标签转换为独热编码
num_classes = len(np.unique(categories))
y_class = tf.keras.utils.to_categorical(y_class, num_classes=num_classes)

# 数据集拆分
X_train, X_test, y_class_train, y_class_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

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

# 模型构建
input_dim = X_train.shape[2]  # 特征数
input_layer = Input(shape=(sequence_length, input_dim))

# 卷积层
cnn_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
cnn_layer = Conv1D(filters=32, kernel_size=3, activation='relu')(cnn_layer)

# LSTM 层
lstm_layer = LSTM(64, return_sequences=True)(cnn_layer)
lstm_layer = LSTM(32, return_sequences=True)(lstm_layer)

# 注意力层
attention_output = AttentionLayer()(lstm_layer)

# 分类预测分支
dense_class = Dense(32, activation='relu')(attention_output)
dense_class = Dropout(0.3)(dense_class)
output_class = Dense(num_classes, activation='softmax', name='class_output')(dense_class)

# 构建模型
model = Model(inputs=input_layer, outputs=output_class)

# 编译模型
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 定义回调函数
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6)

# 训练模型
history = model.fit(
    X_train,
    y_class_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, lr_reduction]
)

# 模型评估
test_loss, test_accuracy = model.evaluate(X_test, y_class_test)
print(f'Test Accuracy: {test_accuracy:.4f}')

# 预测
class_predictions = np.argmax(model.predict(X_test), axis=1)
actual_classes = np.argmax(y_class_test, axis=1)

# 分类报告
print(classification_report(actual_classes, class_predictions))

# 可视化分类结果
plt.figure(figsize=(15, 6))
plt.plot(actual_classes, label='Actual Class', marker='o', color='green', alpha=0.6)
plt.plot(class_predictions, label='Predicted Class', marker='x', linestyle='--', color='red')
plt.title('Actual vs Predicted Classes')
plt.legend()
plt.grid(True)
plt.show()
