import platform
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.display_functions import display
from matplotlib import font_manager
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

# 设置中文字体
def set_chinese_font():
    system = platform.system()
    if system == 'Darwin':  # macOS
        font_path = '/System/Library/Fonts/STHeiti Medium.ttc'
    elif system == 'Windows':
        font_path = 'C:/Windows/Fonts/simhei.ttf'
    else:
        font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
    try:
        font = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font.get_name()
    except:
        print("未找到指定的中文字体，可能导致中文显示异常。")

set_chinese_font()

# 数据加载
file_path = 'orders_2018_2023_v3.csv'
df = pd.read_csv(file_path, encoding='utf-8-sig')

# 数据预处理
df['订单付款时间'] = pd.to_datetime(df['订单付款时间'], format='%Y/%m/%d %H:%M')
df['YearMonth'] = df['订单付款时间'].dt.to_period('M')
monthly_refund = df.groupby('YearMonth').agg({
    '总金额': 'sum',
    '退款金额': 'sum'
}).reset_index()
monthly_refund['退款率'] = monthly_refund['退款金额'] / monthly_refund['总金额']
monthly_refund['YearMonth'] = monthly_refund['YearMonth'].dt.to_timestamp()
monthly_refund.set_index('YearMonth', inplace=True)

# 检查缺失值
monthly_refund = monthly_refund.interpolate(method='linear')
monthly_refund = monthly_refund[monthly_refund['退款率'] <= 1.0]

# 数据可视化
plt.figure(figsize=(14, 7))
sns.scatterplot(x='总金额', y='退款率', data=monthly_refund, color='blue', label='实际值')
plt.title('月总金额 vs 月退款率', fontsize=16)
plt.xlabel('月总金额 (元)', fontsize=14)
plt.ylabel('月退款率', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# 使用Prophet模型预测未来的月总订单金额
prophet_order_df = monthly_refund.reset_index()[['YearMonth', '总金额']]
prophet_order_df.columns = ['ds', 'y']
model_order = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
model_order.fit(prophet_order_df)
future_order = model_order.make_future_dataframe(periods=12, freq='MS')
forecast_order = model_order.predict(future_order)

fig1 = model_order.plot(forecast_order)
plt.title('月总金额预测 (Prophet)', fontsize=16)
plt.xlabel('时间', fontsize=14)
plt.ylabel('月总金额 (元)', fontsize=14)
plt.show()

# 提取2024年的预测结果
forecast_order_2024 = forecast_order[forecast_order['ds'] >= '2024-01-01'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast_order_2024.rename(columns={'yhat': '预测总金额', 'yhat_lower': '下限', 'yhat_upper': '上限'}, inplace=True)
display(forecast_order_2024)

# 回归模型训练与预测退款率
X = monthly_refund[['总金额']]
y = monthly_refund['退款率']
train = monthly_refund[:'2022-12']
test = monthly_refund['2023-01':]

X_train = train[['总金额']]
y_train = train['退款率']
X_test = test[['总金额']]
y_test = test['退款率']

# 线性回归模型
reg = LinearRegression()
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)

mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred)

print("\n线性回归模型评估指标：")
print(f"训练集 - MAE: {mae_train:.4f}, MSE: {mse_train:.4f}, RMSE: {rmse_train:.4f}, R²: {r2_train:.4f}")
print(f"测试集 - MAE: {mae_test:.4f}, MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, R²: {r2_test:.4f}")

# 多项式回归模型
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)

reg_poly = LinearRegression()
reg_poly.fit(X_poly, y_train)
X_test_poly = poly.transform(X_test)
y_test_pred_poly = reg_poly.predict(X_test_poly)

mae_test_poly = mean_absolute_error(y_test, y_test_pred_poly)
mse_test_poly = mean_squared_error(y_test, y_test_pred_poly)
rmse_test_poly = np.sqrt(mse_test_poly)
r2_test_poly = r2_score(y_test, y_test_pred_poly)

print("\n多项式回归模型评估指标：")
print(f"测试集 - MAE: {mae_test_poly:.4f}, MSE: {mse_test_poly:.4f}, RMSE: {rmse_test_poly:.4f}, R²: {r2_test_poly:.4f}")

# 神经网络（MLP）模型
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_nn = Sequential()
model_nn.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
model_nn.add(Dropout(0.2))
model_nn.add(Dense(64, activation='relu'))
model_nn.add(Dropout(0.2))
model_nn.add(Dense(32, activation='relu'))
model_nn.add(Dense(1, activation='linear'))

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model_nn.compile(optimizer='adam', loss='mean_squared_error')
model_nn.fit(X_train_scaled, y_train, epochs=200, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=1, callbacks=[early_stopping])

y_test_pred_nn = model_nn.predict(X_test_scaled)

mae_test_nn = mean_absolute_error(y_test, y_test_pred_nn)
mse_test_nn = mean_squared_error(y_test, y_test_pred_nn)
rmse_test_nn = np.sqrt(mse_test_nn)
r2_test_nn = r2_score(y_test, y_test_pred_nn)

print("\n神经网络模型评估指标：")
print(f"测试集 - MAE: {mae_test_nn:.4f}, MSE: {mse_test_nn:.4f}, RMSE: {rmse_test_nn:.4f}, R²: {r2_test_nn:.4f}")

# LSTM模型
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prophet_order_df['y'].values.reshape(-1, 1))

def create_lstm_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 12
X, y = create_lstm_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model_lstm = Sequential()
model_lstm.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(units=100, return_sequences=True))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(units=50))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(units=1))

model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), verbose=1)

y_test_pred_lstm = model_lstm.predict(X_test)

mae_test_lstm = mean_absolute_error(y_test, y_test_pred_lstm)
mse_test_lstm = mean_squared_error(y_test, y_test_pred_lstm)
rmse_test_lstm = np.sqrt(mse_test_lstm)
r2_test_lstm = r2_score(y_test, y_test_pred_lstm)

print("\nLSTM模型评估指标：")
print(f"测试集 - MAE: {mae_test_lstm:.4f}, MSE: {mse_test_lstm:.4f}, RMSE: {rmse_test_lstm:.4f}, R²: {r2_test_lstm:.4f}")
