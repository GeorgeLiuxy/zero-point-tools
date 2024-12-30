import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# ========================== 配置参数 ==========================
CONFIG = {
    'sequence_length': 10,
    'test_size': 0.2,
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'num_classes': 4,
    'loss_weights': {'value_output': 0.5, 'class_output': 0.5},
    'early_stopping': {'monitor': 'val_loss', 'patience': 10, 'restore_best_weights': True},
    'lr_reduction': {'monitor': 'val_loss', 'patience': 3, 'factor': 0.5, 'min_lr': 1e-6}
}

# ========================= 数据预处理函数 =========================
def load_and_preprocess_data(filepath, config):
    # 读取数据
    df = pd.read_csv(filepath, parse_dates=['Timestamp'], index_col='Timestamp')

    # 提取特征和标签
    values = df['Value'].values
    categories = df['Category'].values

    # 归一化数值数据
    scaler = MinMaxScaler(feature_range=(0, 1))
    values_scaled = scaler.fit_transform(values.reshape(-1, 1))

    # 创建时间序列数据
    X, y_value, y_class = create_sequences_with_class(
        data=values_scaled,
        categories=categories,
        seq_length=config['sequence_length']
    )

    # 转换类别为独热编码
    y_class = tf.keras.utils.to_categorical(y_class, num_classes=config['num_classes'])

    # 拆分训练集和测试集
    X_train, X_test, y_value_train, y_value_test, y_class_train, y_class_test = train_test_split(
        X, y_value, y_class, test_size=config['test_size'], random_state=42
    )

    # 扩展维度以适应模型输入要求
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    return X_train, X_test, y_value_train, y_value_test, y_class_train, y_class_test, scaler

def create_sequences_with_class(data, categories, seq_length):
    """创建时间序列数据，支持类别信息."""
    X, y_value, y_class = [], [], []
    for i in range(len(data) - seq_length):
        features = np.column_stack((data[i:i + seq_length], categories[i:i + seq_length]))
        X.append(features)
        y_value.append(data[i + seq_length])
        y_class.append(categories[i + seq_length])
    return np.array(X), np.array(y_value), np.array(y_class)

# ========================= 模型定义函数 =========================
class AttentionLayer(Layer):
    """自定义注意力层."""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform', trainable=True
        )
        self.b = self.add_weight(
            shape=(input_shape[-1],), initializer='zeros', trainable=True
        )
        self.u = self.add_weight(
            shape=(input_shape[-1],), initializer='glorot_uniform', trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        u_t = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        a_t = tf.nn.softmax(tf.tensordot(u_t, self.u, axes=1), axis=1)
        output = inputs * tf.expand_dims(a_t, -1)
        return tf.reduce_sum(output, axis=1)

def build_model(config):
    """构建多任务模型."""
    input_layer = Input(shape=(config['sequence_length'], 2))

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
    output_class = Dense(config['num_classes'], activation='softmax', name='class_output')(dense_class)

    # 构建模型
    model = Model(inputs=input_layer, outputs=[output_value, output_class])
    model.compile(
        optimizer=Adam(learning_rate=config['learning_rate']),
        loss={'value_output': 'mse', 'class_output': 'categorical_crossentropy'},
        loss_weights=config['loss_weights']
    )
    return model

# ========================= 训练和评估函数 =========================
def train_model(model, X_train, y_train, config):
    """训练模型."""
    callbacks = [
        EarlyStopping(**config['early_stopping']),
        ReduceLROnPlateau(**config['lr_reduction'])
    ]
    history = model.fit(
        X_train,
        y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    return history

def evaluate_model(model, X_test, y_test, scaler):
    """评估模型性能."""
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    predictions = model.predict(X_test)

    # 数值预测
    value_predictions = scaler.inverse_transform(predictions[0])
    actual_values = scaler.inverse_transform(y_test['value_output'].reshape(-1, 1))

    # 分类预测
    class_predictions = np.argmax(predictions[1], axis=1)
    actual_classes = np.argmax(y_test['class_output'], axis=1)

    # 计算指标
    mae = mean_absolute_error(actual_values, value_predictions)
    mse = mean_squared_error(actual_values, value_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_values, value_predictions)
    accuracy = np.mean(class_predictions == actual_classes)

    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, Accuracy: {accuracy:.4f}")
    return value_predictions, class_predictions, actual_values, actual_classes

def plot_results(history, actual_values, value_predictions, actual_classes, class_predictions):
    """绘制结果，包括训练损失、数值预测和分类预测对比."""
    plt.figure(figsize=(15, 10))

    # 数值预测对比
    plt.subplot(3, 1, 1)
    plt.plot(actual_values, label='Actual Value', marker='o', color='blue', alpha=0.6)
    plt.plot(value_predictions, label='Predicted Value', marker='x', linestyle='--', color='orange')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.grid(True)

    # 分类预测对比
    plt.subplot(3, 1, 2)
    plt.plot(actual_classes, label='Actual Class', marker='o', color='green', alpha=0.6)
    plt.plot(class_predictions, label='Predicted Class', marker='x', linestyle='--', color='red')
    plt.title('Actual vs Predicted Classes')
    plt.legend()
    plt.grid(True)

    # 训练过程损失变化
    plt.subplot(3, 1, 3)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
def preprocess_new_data(filepath, scaler, seq_length):
    """预处理新数据文件."""
    df = pd.read_csv(filepath, parse_dates=['Timestamp'], index_col='Timestamp')
    values = df['Value'].values
    categories = df['Category'].values

    # 使用训练时的Scaler对数值进行归一化
    values_scaled = scaler.transform(values.reshape(-1, 1))

    # 创建时间序列数据
    X, _, _ = create_sequences_with_class(
        data=values_scaled,
        categories=categories,
        seq_length=seq_length
    )

    # 调整数据形状以适应模型输入要求
    X = np.expand_dims(X, axis=-1)
    return X, df.iloc[seq_length:]  # 返回时间序列特征和对应的原始数据（对齐时间戳）

def predict_new_data(model, X, scaler, label_encoder=None):
    """使用训练好的模型对新数据进行预测."""
    predictions = model.predict(X)

    # 数值预测（反归一化）
    value_predictions = scaler.inverse_transform(predictions[0])

    # 分类预测（解码为类别标签）
    class_predictions = np.argmax(predictions[1], axis=1)
    if label_encoder:
        class_predictions = label_encoder.inverse_transform(class_predictions)

    return value_predictions, class_predictions

def save_predictions_to_csv(filepath, df, value_preds, class_preds):
    """保存预测结果到CSV文件."""
    df['Predicted_Value'] = value_preds.flatten()
    df['Predicted_Class'] = class_preds
    output_path = filepath.replace('.csv', '_predictions.csv')
    df.to_csv(output_path)
    print(f"Predictions saved to {output_path}")

# ========================= 主流程（预测新数据） =========================
if __name__ == "__main__":
    # 加载和预处理训练数据
    train_filepath = './cnn_data/10294.csv'
    X_train, X_test, y_value_train, y_value_test, y_class_train, y_class_test, scaler = load_and_preprocess_data(train_filepath, CONFIG)

    # 构建并训练模型
    model = build_model(CONFIG)
    history = train_model(
        model,
        X_train,
        {'value_output': y_value_train, 'class_output': y_class_train},
        CONFIG
    )

    # 保存训练好的模型
    model.save('trained_model.keras')

    # 评估模型
    value_preds, class_preds, actual_values, actual_classes = evaluate_model(
        model,
        X_test,
        {'value_output': y_value_test, 'class_output': y_class_test},
        scaler
    )

    # 绘制训练结果
    plot_results(history, actual_values, value_preds, actual_classes, class_preds)

    # ==================== 预测新数据 ====================
    # 加载训练好的模型
    trained_model = tf.keras.models.load_model('trained_model.keras', custom_objects={'AttentionLayer': AttentionLayer})

    # 新数据文件路径
    new_data_filepath = './cnn_data/59020.csv'  # 替换为您的新数据文件路径

    # 预处理新数据
    X_new, df_new = preprocess_new_data(new_data_filepath, scaler, CONFIG['sequence_length'])

    # 使用模型对新数据进行预测
    value_preds_new, class_preds_new = predict_new_data(trained_model, X_new, scaler)

    # 保存预测结果
    save_predictions_to_csv(new_data_filepath, df_new, value_preds_new, class_preds_new)

    # 可视化新数据的预测结果
    plt.figure(figsize=(15, 7))

    # 数值预测对比
    plt.subplot(2, 1, 1)
    plt.plot(df_new.index, df_new['Value'], label='Actual Value', marker='o', color='blue', alpha=0.6)
    plt.plot(df_new.index, value_preds_new.flatten(), label='Predicted Value', marker='x', linestyle='--', color='orange')
    plt.title('Actual vs Predicted Values on New Data')
    plt.legend()
    plt.grid(True)

    # 分类预测对比
    plt.subplot(2, 1, 2)
    plt.plot(df_new.index, df_new['Category'], label='Actual Class', marker='o', color='green', alpha=0.6)
    plt.plot(df_new.index, class_preds_new, label='Predicted Class', marker='x', linestyle='--', color='red')
    plt.title('Actual vs Predicted Classes on New Data')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
