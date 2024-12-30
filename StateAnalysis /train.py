import re
from datetime import datetime
import pandas as pd
import numpy as np
import os

import pca
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.utils import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from matplotlib import rcParams

# 设置字体
rcParams['font.sans-serif'] = ['Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# --------------------- 数据加载与预处理模块 --------------------- #
def load_and_merge_csv(folder_path):
    """加载并合并CSV文件，添加卫星标识列"""
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not all_files:
        raise FileNotFoundError(f"目录 {folder_path} 中未找到CSV文件。")
    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(os.path.join(folder_path, file)).assign(卫星=os.path.splitext(file)[0])
            df_list.append(df)
        except Exception as e:
            print(f"警告: 文件 {file} 加载失败: {e}")
    return pd.concat(df_list, ignore_index=True)

def parse_date_by_pattern(date_str):
    """根据日期格式解析字符串"""
    pattern_1 = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
    pattern_2 = r"^\d{4}/\d{1,2}/\d{1,2} \d{1,2}:\d{2}$"
    try:
        if re.match(pattern_1, date_str):
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        elif re.match(pattern_2, date_str):
            return datetime.strptime(date_str, "%Y/%m/%d %H:%M")
    except Exception as e:
        print(f"日期解析失败: {date_str} - {e}")
    return None

def preprocess_data(data):
    """数据清洗、特征工程和处理缺失值"""
    data.drop_duplicates(inplace=True)
    data['时间'] = data['时间'].apply(parse_date_by_pattern)
    data = data.dropna(subset=['时间'])  # 丢弃无法解析的日期行
    data['小时'] = data['时间'].dt.hour
    data['星期'] = data['时间'].dt.weekday
    data['月份'] = data['时间'].dt.month
    data['时间戳'] = data['时间'].astype(np.int64) // 10**9

    def get_season(month):
        return ['冬季', '春季', '夏季', '秋季'][(month % 12 + 3) // 3 - 1]

    data['季节'] = data['月份'].apply(get_season)
    data = pd.get_dummies(data, columns=['季节'], drop_first=True)

    for col, strategy in {'半长轴': 'median', '星下点经度(°)': 'median', '控制状态': 'mode'}.items():
        if col in data:
            if strategy == 'median':
                data[col] = data[col].fillna(data[col].median())
            elif strategy == 'mode':
                data[col] = data[col].fillna(data[col].mode()[0])

    def remove_outliers_iqr(df, column):
        if column in df:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df

    data = remove_outliers_iqr(data, '半长轴')
    return data

# --------------------- 特征工程模块 --------------------- #
def feature_engineering(data):
    """特征构建和处理，包括编码和标准化"""
    if '控制状态' not in data:
        raise KeyError("数据缺少 '控制状态' 列，无法进行标签编码。")

    label_encoder = LabelEncoder()
    data['控制状态_encoded'] = label_encoder.fit_transform(data['控制状态'])

    numerical_features = ['半长轴', '星下点经度(°)', '异常值', '时间戳', '小时', '星期', '月份']
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    data['半长轴_经度'] = data['半长轴'] * data['星下点经度(°)']

    return data, label_encoder, scaler, numerical_features

# --------------------- PCA降维模块 --------------------- #
def apply_pca(data, numerical_features, pca_model=None):
    """
    对数据进行 PCA 降维
    """
    # 检查是否生成了必须的列
    # # 确保列名匹配
    # required_columns = numerical_features + ['半长轴_经度']
    # missing_columns = [col for col in required_columns if col not in new_data.columns]
    # extra_columns = [col for col in new_data.columns if col not in required_columns]
    # if missing_columns:
    #     print("新数据缺少以下列:", missing_columns)
    # if extra_columns:
    #     print("新数据包含多余列:", extra_columns)

    # 检查是否生成了必须的列
    required_columns = numerical_features + ['半长轴_经度']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise KeyError(f"数据缺少以下列: {missing_columns}")


    # 执行 PCA 降维
    if pca_model is None:
        pca = PCA(n_components=0.95, random_state=42)
        principal_components = pca.fit_transform(data[required_columns])
        joblib.dump(pca, 'pca_model.pkl')  # 保存 PCA 模型
    else:
        principal_components = pca_model.transform(data[required_columns])

    pca_columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
    return pd.DataFrame(principal_components, columns=pca_columns)


# --------------------- 时序数据构建模块 --------------------- #
def create_sequences(df, features, label, time_steps=20):
    """创建时序数据"""
    X, y = [], []
    for satellite in df['卫星'].unique():
        satellite_data = df[df['卫星'] == satellite].sort_values('时间')
        data_values = satellite_data[features].values
        label_values = satellite_data[label].values
        for i in range(len(data_values) - time_steps):
            X.append(data_values[i:i+time_steps])
            y.append(label_values[i+time_steps])
    return np.array(X), np.array(y)

# --------------------- 模型构建模块 --------------------- #
def build_cnn_lstm_attention_model(input_shape, num_classes):
    """构建CNN-LSTM-Attention模型"""
    inputs = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=3, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    x = Conv1D(64, kernel_size=3, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = LSTM(64, return_sequences=True)(x)
    attention = Attention()([x, x])
    attention = GlobalAveragePooling1D()(attention)
    x = Dense(64, activation='relu')(attention)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid')(x)

    return Model(inputs=inputs, outputs=outputs)

# --------------------- 特征对齐模块 --------------------- #
def adjust_features(X_new, X_train):
    """调整新数据特征数以匹配训练数据"""
    expected_features = X_train.shape[2]
    actual_features = X_new.shape[2]
    if actual_features != expected_features:
        print(f"特征数不一致：期望 {expected_features}，实际 {actual_features}")
        if actual_features > expected_features:
            X_new = X_new[:, :, :expected_features]  # 截断多余的特征
        else:
            padding = np.zeros((X_new.shape[0], X_new.shape[1], expected_features - actual_features))
            X_new = np.concatenate([X_new, padding], axis=2)
    return X_new

def preprocess_new_data(new_folder_path, scaler, label_encoder, time_steps=20):
    """
    新数据预处理和序列构建
    """
    # 加载并预处理新数据
    new_data = load_and_merge_csv(new_folder_path)
    new_data = preprocess_data(new_data)

    # 使用训练时的标准化器进行标准化
    numerical_features = ['半长轴', '星下点经度(°)', '异常值', '时间戳', '小时', '星期', '月份']
    new_data[numerical_features] = scaler.transform(new_data[numerical_features])

    # 确保生成新的特征: 半长轴_经度
    if '半长轴' in new_data and '星下点经度(°)' in new_data:
        new_data['半长轴_经度'] = new_data['半长轴'] * new_data['星下点经度(°)']
    else:
        raise KeyError("新数据中缺少 '半长轴' 或 '星下点经度(°)' 列，无法生成 '半长轴_经度' 特征。")

    # 检查并处理 '控制状态' 列
    if '控制状态' in new_data:
        new_labels = np.unique(new_data['控制状态'])
        known_labels = set(label_encoder.classes_)
        unknown_labels = set(new_labels) - known_labels

        # 检查是否存在未知类别
        if unknown_labels:
            raise ValueError(f"新数据中存在未知的 '控制状态' 类别: {unknown_labels}")

        # 编码 '控制状态'
        new_data['控制状态_encoded'] = label_encoder.transform(new_data['控制状态'])
    else:
        raise KeyError("新数据中缺少 '控制状态' 列，无法生成 '控制状态_encoded'。")

    # 加载 PCA 模型并进行降维
    pca = joblib.load('pca_model.pkl')
    required_columns = numerical_features + ['半长轴_经度']
    missing_columns = [col for col in required_columns if col not in new_data.columns]
    if missing_columns:
        raise KeyError(f"新数据缺少以下列: {missing_columns}")
    pca_df = pd.DataFrame(pca.transform(new_data[required_columns]),
                          columns=[f'PC{i+1}' for i in range(pca.n_components_)])

    # 合并降维后的数据
    new_data = pd.concat([new_data, pca_df], axis=1)

    # 构建时序数据
    features = numerical_features + list(pca_df.columns)
    X_new, _ = create_sequences(new_data, features, '控制状态_encoded', time_steps)

    return X_new, new_data


# --------------------- 主流程 --------------------- #
if __name__ == "__main__":
    # 数据加载与预处理
    folder_path = 'train_data'
    data = load_and_merge_csv(folder_path)
    data = preprocess_data(data)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()  # 清理异常值
    data, label_encoder, scaler, numerical_features = feature_engineering(data)

    # PCA降维
    pca_df = apply_pca(data, numerical_features)
    data = pd.concat([data, pca_df], axis=1)

    # 构建时序数据
    time_steps = 20
    features = numerical_features + list(pca_df.columns)
    label = '控制状态_encoded'
    X, y = create_sequences(data, features, label, time_steps)

    # 数据分割
    split1, split2 = int(0.7 * len(X)), int(0.85 * len(X))
    X_train, X_val, X_test = X[:split1], X[split1:split2], X[split2:]
    y_train, y_val, y_test = y[:split1], y[split1:split2], y[split2:]

    # 模型构建与训练
    model = build_cnn_lstm_attention_model(X_train.shape[1:], len(label_encoder.classes_))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
    ]

    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    # 类别权重缩放
    max_weight = 10
    class_weights = {cls: min(weight, max_weight) for cls, weight in enumerate(class_weights_array)}

    print("类别权重:", class_weights)

    # 在模型训练时使用
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_val, y_val),
        class_weight=class_weights,  # 加入类别权重
        callbacks=callbacks
    )
    # 评估模型
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"测试集损失: {test_loss}, 测试集准确率: {test_accuracy}")

    # 混淆矩阵与分类报告
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred, labels=range(len(label_encoder.classes_)))
    # 修改分类报告代码部分
    report = classification_report(
        y_test, y_pred,
        target_names=[str(c) for c in label_encoder.classes_],
        zero_division=0  # 避免未定义的精确度警告
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.savefig('confusion_matrix.png')

    with open("classification_report.txt", "w") as f:
        f.write(report)

    model.save('satellite_control_model.keras')

    # 新数据处理与预测
    new_folder_path = 'new_satellite_data'
    pca_model = joblib.load('pca_model.pkl')
    new_data = load_and_merge_csv(new_folder_path)
    X_new, new_data = preprocess_new_data(new_folder_path, scaler, label_encoder, time_steps)
    # final_features = numerical_features + ['半长轴_经度'] + [f'PC{i+1}' for i in range(pca.n_components_)]
    # new_data = new_data[final_features]  # 确保新数据特征与训练数据一致
    new_data[numerical_features] = scaler.transform(new_data[numerical_features])
    pca_df = apply_pca(new_data, numerical_features, pca_model=pca_model)
    new_data = pd.concat([new_data, pca_df], axis=1)

    X_new, _ = create_sequences(new_data, features, label, time_steps)
    X_new = adjust_features(X_new, X_train)

    y_new_pred = np.argmax(model.predict(X_new), axis=1)
    new_data = new_data.iloc[time_steps:].reset_index(drop=True)
    new_data['预测控制状态'] = label_encoder.inverse_transform(y_new_pred)
    new_data.to_csv('predicted_control_states.csv', index=False)
    print("预测结果已保存到 'predicted_control_states.csv'")
