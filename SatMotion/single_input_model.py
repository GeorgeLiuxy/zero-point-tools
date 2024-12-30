import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score
)
import tensorflow as tf
from tensorflow.keras.models import Model, load_model

from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy

from tensorflow.keras.layers import (
    Input,
    Conv1D,
    LSTM,
    Dense,
    Dropout,
    Layer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 配置字典
CONFIG = {
    'data_path': Path('./merged_data.csv'),
    'sequence_length': 10,
    'test_size': 0.2,
    'random_state': 42,
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'loss_weights': {'value_output': 0.5, 'class_output': 0.5},
    'callbacks': {
        'early_stopping': {'monitor': 'val_loss', 'patience': 10, 'restore_best_weights': True},
        'lr_reduction': {'monitor': 'val_loss', 'patience': 3, 'factor': 0.5, 'min_lr': 1e-6}
    }
}

def create_sequences(data, categories, seq_length):
    X, y_value, y_class = [], [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y_value.append(data[i + seq_length])
        y_class.append(categories[i + seq_length])
    return np.array(X), np.array(y_value), np.array(y_class)

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True,
            name='W'
        )
        self.b = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True,
            name='b'
        )
        self.u = self.add_weight(
            shape=(input_shape[-1],),
            initializer='glorot_uniform',
            trainable=True,
            name='u'
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # 计算 u_t
        u_t = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        # 计算注意力权重 a_t
        a_t = tf.nn.softmax(tf.tensordot(u_t, self.u, axes=1), axis=1)
        # 计算加权和
        output = inputs * tf.expand_dims(a_t, -1)
        return tf.reduce_sum(output, axis=1)

def load_and_preprocess_data(config):
    # 读取 CSV 数据
    df = pd.read_csv(config['data_path'], parse_dates=['Timestamp'], index_col='Timestamp')

    # 提取数值和类别列
    values = df['Value'].values
    categories = df['Category'].values

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    values_scaled = scaler.fit_transform(values.reshape(-1, 1))

    # 创建时间序列数据
    X, y_value, y_class = create_sequences(
        data=values_scaled,
        categories=categories,
        seq_length=config['sequence_length']
    )

    # 使用 LabelEncoder 对类别标签进行编码
    label_encoder = LabelEncoder()
    y_class = label_encoder.fit_transform(y_class)

    # 自动确定类别数量
    num_classes = len(label_encoder.classes_)

    # 将类别标签转换为独热编码
    y_class = tf.keras.utils.to_categorical(y_class, num_classes=num_classes)

    # 数据集拆分
    X_train, X_test, y_value_train, y_value_test, y_class_train, y_class_test = train_test_split(
        X, y_value, y_class, test_size=config['test_size'], random_state=config['random_state']
    )

    # 调整数据形状以适应 CNN 和 LSTM 的输入要求
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    return (
        X_train, X_test,
        y_value_train, y_value_test,
        y_class_train, y_class_test,
        scaler,
        num_classes
    )

def build_model(sequence_length, num_classes):
    input_layer = Input(shape=(sequence_length, 1), name='input_layer')

    # 卷积层
    cnn = Conv1D(filters=64, kernel_size=3, activation='relu', name='conv1')(input_layer)
    cnn = Conv1D(filters=32, kernel_size=3, activation='relu', name='conv2')(cnn)

    # LSTM 层
    lstm = LSTM(64, return_sequences=True, name='lstm1')(cnn)
    lstm = LSTM(32, return_sequences=True, name='lstm2')(lstm)

    # 注意力层
    attention = AttentionLayer(name='attention')(lstm)

    # 数值预测分支
    value_branch = Dense(32, activation='relu', name='dense_value_1')(attention)
    value_branch = Dropout(0.3, name='dropout_value')(value_branch)
    output_value = Dense(1, name='value_output')(value_branch)

    # 分类预测分支
    class_branch = Dense(32, activation='relu', name='dense_class_1')(attention)
    class_branch = Dropout(0.3, name='dropout_class')(class_branch)
    output_class = Dense(num_classes, activation='softmax', name='class_output')(class_branch)

    # 构建模型
    model = Model(inputs=input_layer, outputs=[output_value, output_class], name='MultiTask_Model')

    return model

def compile_model(model, config):
    model.compile(
        optimizer=Adam(learning_rate=config['learning_rate']),
        loss={
            'value_output': 'mse',
            'class_output': 'categorical_crossentropy'
        },
        loss_weights=config['loss_weights'],
        metrics={
            'value_output': ['mae', 'mse'],
            'class_output': ['accuracy']
        }
    )
    return model

def train_model(model, X_train, y_train, config):
    callbacks = [
        EarlyStopping(**config['callbacks']['early_stopping']),
        ReduceLROnPlateau(**config['callbacks']['lr_reduction'])
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
    # 模型评估
    test_loss, test_value_loss, test_class_loss, test_value_mae, test_value_mse, test_class_acc = model.evaluate(
        X_test, y_test, verbose=0
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Value Loss (MSE): {test_value_loss:.4f}")
    print(f"Test Class Loss (Categorical Crossentropy): {test_class_loss:.4f}")
    print(f"Test Value MAE: {test_value_mae:.4f}")
    print(f"Test Value MSE: {test_value_mse:.4f}")
    print(f"Test Class Accuracy: {test_class_acc:.4f}")

def make_predictions(model, X_test, scaler):
    predictions = model.predict(X_test)
    value_predictions = scaler.inverse_transform(predictions[0])
    class_predictions = np.argmax(predictions[1], axis=1)
    return value_predictions, class_predictions

def plot_results(y_test, value_predictions, class_predictions, scaler, history):
    actual_values = scaler.inverse_transform(y_test['value_output'].reshape(-1, 1))
    actual_classes = np.argmax(y_test['class_output'], axis=1)

    plt.figure(figsize=(15, 15))

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

    # 训练过程中的损失变化
    plt.subplot(3, 1, 3)
    plt.plot(history.history['loss'], label='Training Loss', color='purple')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='brown')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def calculate_metrics(y_test, value_predictions, class_predictions, scaler):
    actual_values = scaler.inverse_transform(y_test['value_output'].reshape(-1, 1))
    actual_classes = np.argmax(y_test['class_output'], axis=1)

    # 数值预测指标
    mae = mean_absolute_error(actual_values, value_predictions)
    mse = mean_squared_error(actual_values, value_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_values, value_predictions)

    # 分类预测指标
    accuracy = accuracy_score(actual_classes, class_predictions)

    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    print(f'R² Score: {r2:.4f}')
    print(f'Classification Accuracy: {accuracy:.4f}')

def save_model(model, filepath='multi_task_model.h5'):
    model.save(filepath)

def load_trained_model(filepath='multi_task_model.h5'):
    trained_model = load_model(filepath, custom_objects={'AttentionLayer': AttentionLayer}, compile=False)

    # 手动编译模型
    trained_model.compile(
        optimizer=Adam(learning_rate=CONFIG['learning_rate']),
        loss={
            'value_output': MeanSquaredError(),
            'class_output': CategoricalCrossentropy()
        },
        loss_weights=CONFIG['loss_weights'],
        metrics={
            'value_output': ['mae', 'mse'],
            'class_output': ['accuracy']
        }
    )
    return trained_model

def preprocess_new_data(new_data_path, scaler, seq_length):
    df_new = pd.read_csv(new_data_path, parse_dates=['Timestamp'], index_col='Timestamp')
    values_new = df_new['Value'].values
    categories_new = df_new['Category'].values

    values_scaled_new = scaler.transform(values_new.reshape(-1, 1))
    X_new, _, _ = create_sequences(
        data=values_scaled_new,
        categories=categories_new,
        seq_length=seq_length
    )
    X_new = np.expand_dims(X_new, axis=-1)
    return X_new

def predict_new_data(model, X_new, scaler, label_encoder=None):
    predictions = model.predict(X_new)
    value_predictions = scaler.inverse_transform(predictions[0])
    class_predictions = np.argmax(predictions[1], axis=1)
    if label_encoder is not None:
        class_predictions = label_encoder.inverse_transform(class_predictions)
    return value_predictions, class_predictions

if __name__ == "__main__":
    # 加载和预处理数据
    X_train, X_test, y_value_train, y_value_test, y_class_train, y_class_test, scaler, num_classes = load_and_preprocess_data(CONFIG)

    # 构建模型
    model = build_model(sequence_length=CONFIG['sequence_length'], num_classes=num_classes)
    model.summary()

    # 编译模型
    model = compile_model(model, CONFIG)

    # 训练模型
    history = train_model(
        model,
        X_train,
        {'value_output': y_value_train, 'class_output': y_class_train},
        CONFIG
    )

    # 保存模型
    save_model(model, 'multi_task_model.h5')

    # 模型评估
    evaluate_model(
        model,
        X_test,
        {'value_output': y_value_test, 'class_output': y_class_test},
        scaler
    )

    # 预测
    value_preds, class_preds = make_predictions(model, X_test, scaler)

    # 绘制结果
    plot_results(
        {'value_output': y_value_test, 'class_output': y_class_test},
        value_preds,
        class_preds,
        scaler,
        history
    )

    # 计算评估指标
    calculate_metrics(
        {'value_output': y_value_test, 'class_output': y_class_test},
        value_preds,
        class_preds,
        scaler
    )

    # ================== 新增部分 ==================

    # 加载新数据的路径
    new_data_path = Path('./cnn_data/59020.csv')  # 替换为您的新数据文件路径

    # 预处理新数据
    X_new = preprocess_new_data(
        new_data_path=new_data_path,
        scaler=scaler,
        seq_length=CONFIG['sequence_length']
    )

    # 加载已训练的模型
    trained_model = load_trained_model('multi_task_model.h5')

    # 进行预测
    value_preds_new, class_preds_new = predict_new_data(
        model=trained_model,
        X_new=X_new,
        scaler=scaler
        # 如果有 label_encoder，可以传入
    )

    # 读取新数据的时间戳
    df_new = pd.read_csv(new_data_path, parse_dates=['Timestamp'], index_col='Timestamp')
    df_new = df_new.iloc[CONFIG['sequence_length']:]  # 对应的时间戳

    # 创建预测结果的 DataFrame
    df_new_predictions = pd.DataFrame({
        'Timestamp': df_new.index,
        'Actual_Value': df_new['Value'].values,
        'Predicted_Value': value_preds_new.flatten(),
        'Actual_Class': df_new['Category'].values,
        'Predicted_Class': class_preds_new
    })

    # 保存预测结果
    df_new_predictions.to_csv('new_data_predictions.csv', index=False)

    # 可视化新数据的预测结果
    plt.figure(figsize=(15, 7))

    # 数值预测对比
    plt.subplot(2, 1, 1)
    plt.plot(df_new_predictions['Timestamp'], df_new_predictions['Actual_Value'], label='Actual Value', marker='o', color='blue', alpha=0.6)
    plt.plot(df_new_predictions['Timestamp'], df_new_predictions['Predicted_Value'], label='Predicted Value', marker='x', linestyle='--', color='orange')
    plt.title('Actual vs Predicted Values on New Data')
    plt.legend()
    plt.grid(True)

    # 分类预测对比
    plt.subplot(2, 1, 2)
    plt.plot(df_new_predictions['Timestamp'], df_new_predictions['Actual_Class'], label='Actual Class', marker='o', color='green', alpha=0.6)
    plt.plot(df_new_predictions['Timestamp'], df_new_predictions['Predicted_Class'], label='Predicted Class', marker='x', linestyle='--', color='red')
    plt.title('Actual vs Predicted Classes on New Data')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
