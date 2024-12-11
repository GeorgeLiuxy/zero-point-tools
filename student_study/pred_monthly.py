# 导入相关库
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
# 线性回归

# 设置中文字体
def set_chinese_font():
    """
    设置Matplotlib的中文字体，以确保图表中的中文能够正确显示。
    根据操作系统选择字体路径。
    """
    system = platform.system()
    if system == 'Darwin':  # macOS
        font_path = '/System/Library/Fonts/STHeiti Medium.ttc'
    elif system == 'Windows':
        font_path = 'C:/Windows/Fonts/simhei.ttf'  # 常见的Windows中文字体
    else:
        # 如果是Linux或其他系统，可以尝试使用以下路径或安装相应字体
        font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'  # 示例路径
    try:
        font = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font.get_name()
    except:
        print("未找到指定的中文字体，可能导致中文显示异常。")

set_chinese_font()

# 数据加载
file_path = 'orders_2018_2023_v3.csv'  # 请确保文件路径正确
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

# 查看聚合后的数据
print("聚合后的数据预览：")
display(monthly_refund.head())

# 检查缺失值
print("\n缺失值情况：")
display(monthly_refund.isnull().sum())

monthly_refund = monthly_refund.dropna()

# 退款率统计信息
print("\n退款率统计信息：")
display(monthly_refund['退款率'].describe())

# 剔除退款率大于1的数据
monthly_refund = monthly_refund[monthly_refund['退款率'] <= 1.0]

# 3. 数据可视化 - 月总金额与月退款率的关系
plt.figure(figsize=(14, 7))
sns.scatterplot(x='总金额', y='退款率', data=monthly_refund, color='blue', label='实际值')
plt.title('月总金额 vs 月退款率', fontsize=16)
plt.xlabel('月总金额 (元)', fontsize=14)
plt.ylabel('月退款率', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# 图表解释：
# 上图展示了每月的总金额与退款率之间的关系。我们可以看到，当月总金额较大时，退款率相对较低，
# 这可能表明退款率与订单金额的大小存在一定的反向关系。

# 4. 使用Prophet模型预测未来的月总订单金额
prophet_order_df = monthly_refund.reset_index()[['YearMonth', '总金额']]
prophet_order_df.columns = ['ds', 'y']
model_order = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
model_order.fit(prophet_order_df)
future_order = model_order.make_future_dataframe(periods=12, freq='MS')
forecast_order = model_order.predict(future_order)

# 可视化订单总金额的预测
fig1 = model_order.plot(forecast_order)
plt.title('月总金额预测 (Prophet)', fontsize=16)
plt.xlabel('时间', fontsize=14)
plt.ylabel('月总金额 (元)', fontsize=14)
plt.show()

# 图表解释：
# 上图展示了使用Prophet模型对未来12个月的月总金额进行的预测。红色的线代表历史数据的趋势，
# 绿色的带状区域表示预测的可信区间。可以看到，未来几个月的订单总金额呈现出一定的增长趋势。

# 提取2024年的预测结果
forecast_order_2024 = forecast_order[forecast_order['ds'] >= '2024-01-01'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast_order_2024.rename(columns={'yhat': '预测总金额', 'yhat_lower': '下限', 'yhat_upper': '上限'}, inplace=True)
print("\n2024年每月订单总金额预测：")
display(forecast_order_2024)

# 5. 回归模型训练与预测退款率
# 定义自变量（X）和因变量（y）
X = monthly_refund[['总金额']]
y = monthly_refund['退款率']

# 拆分训练集和测试集（2018-2022年作为训练集，2023年作为测试集）
train = monthly_refund[:'2022-12']
test = monthly_refund['2023-01':]

X_train = train[['总金额']]
y_train = train['退款率']
X_test = test[['总金额']]
y_test = test['退款率']

# 初始化并训练线性回归模型
reg = LinearRegression()
reg.fit(X_train, y_train)

# 预测
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

# 6. 模型评估
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)

mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred)

print("\n模型评估指标：")
print("训练集 - MAE: {:.4f}, MSE: {:.4f}, RMSE: {:.4f}, R²: {:.4f}".format(mae_train, mse_train, rmse_train, r2_train))
print("测试集 - MAE: {:.4f}, MSE: {:.4f}, RMSE: {:.4f}, R²: {:.4f}".format(mae_test, mse_test, rmse_test, r2_test))

# 7. 预测未来退款率
# 提取2024年的预测订单总金额
future_order_2024 = forecast_order_2024[['ds', '预测总金额']].copy()
future_order_2024.rename(columns={'ds': 'YearMonth'}, inplace=True)

# 修正：确保特征名称与训练时一致
future_order_2024.rename(columns={'预测总金额': '总金额'}, inplace=True)

# 进行退款率预测
future_refund_rate_pred = reg.predict(future_order_2024[['总金额']])

# 创建预测结果DataFrame
future_refund_df = pd.DataFrame({
    'YearMonth': future_order_2024['YearMonth'],
    '预测总金额': forecast_order_2024['预测总金额'],
    '退款率预测': future_refund_rate_pred
})
print("\n2024年每月退款率预测：")
display(future_refund_df)

# 8. 绘制回归曲线与预测结果
# 预测所有数据的退款率（用于绘制回归线）
monthly_refund_sorted = monthly_refund.sort_values(by='总金额')
sorted_X = monthly_refund_sorted[['总金额']]
sorted_y_pred = reg.predict(sorted_X)

plt.figure(figsize=(14, 7))
# 实际值
sns.scatterplot(x='总金额', y='退款率', data=monthly_refund, color='blue', label='实际值')

# 回归线
plt.plot(monthly_refund_sorted['总金额'], sorted_y_pred, color='red', label='回归线')

# 训练集预测
plt.scatter(X_train, y_train_pred, color='green', label='训练集预测', alpha=0.6)

# 测试集预测
plt.scatter(X_test, y_test_pred, color='orange', label='测试集预测', alpha=0.6)

# 未来预测
sns.scatterplot(x='预测总金额', y='退款率预测', data=future_refund_df, color='purple', label='未来预测', s=100)

plt.title('月总金额 vs 月退款率 - 回归分析与预测', fontsize=16)
plt.xlabel('月总金额 (元)', fontsize=14)
plt.ylabel('月退款率', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# 图表解释：
# 上图展示了基于总金额的退款率回归分析。图中蓝色的点表示实际的退款率与总金额的关系，
# 红色的回归线展示了这一关系的线性拟合结果。绿色和橙色的点分别代表训练集和测试集的预测结果，
# 紫色点则表示基于未来订单预测的退款率。
