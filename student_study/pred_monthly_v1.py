import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib
import os
from matplotlib.ticker import MaxNLocator

# 1. 配置 matplotlib 使用支持中文的字体
# 查找系统中可用的中文字体，例如 SimHei 或 Microsoft YaHei
# 你可以根据自己的系统调整字体路径和名称

# 示例中尝试使用 SimHei
# 如果你使用的是 macOS 或其他系统，字体名称可能不同

# 查找系统中可用的字体
from matplotlib.font_manager import FontProperties

# 设置字体路径（根据你的系统和字体进行调整）
# 例如，macOS 常见中文字体路径：
# '/System/Library/Fonts/STHeiti Light.ttc'
# Windows 常见中文字体路径：
# 'C:/Windows/Fonts/simhei.ttf'
# 你需要根据自己的系统修改路径

# 尝试使用 SimHei
font_path = ""
if os.name == 'nt':  # Windows
    font_path = "C:/Windows/Fonts/simhei.ttf"
elif os.name == 'posix':  # macOS 或 Linux
    # macOS 示例
    font_path = "/System/Library/Fonts/STHeiti Light.ttc"
    # 如果是 Linux，可能需要安装并指定路径，如 '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
else:
    raise RuntimeError("Unsupported OS")

# 检查字体文件是否存在
if not os.path.exists(font_path):
    raise FileNotFoundError(f"指定的字体文件未找到: {font_path}")

# 创建 FontProperties 对象
font_prop = FontProperties(fname=font_path)

# 更新 matplotlib 的字体配置
matplotlib.rcParams['font.family'] = font_prop.get_name()
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 2. 读取已生成的订单数据
df = pd.read_csv("orders_2018_2023.csv", encoding='utf-8-sig')

# 3. 处理时间数据
df['订单创建时间'] = pd.to_datetime(df['订单创建时间'])

# 提取年、月信息
df['年'] = df['订单创建时间'].dt.year
df['月'] = df['订单创建时间'].dt.month

# 4. 计算每月的退款率
df['有退款'] = df['退款金额'].apply(lambda x: 1 if x > 0 else 0)

monthly_refund_data = df.groupby(['年', '月']).agg(
    total_orders=('订单编号', 'count'),
    refunded_orders=('有退款', 'sum'),
    total_sales=('总金额', 'sum')  # 添加每月总销售额
).reset_index()

monthly_refund_data['退款率'] = monthly_refund_data['refunded_orders'] / monthly_refund_data['total_orders']

# 打印前几行数据进行检查
print(monthly_refund_data.head())

# 5. 销售额回归分析：使用线性回归模型预测销售额
monthly_refund_data['时间'] = monthly_refund_data['年'] * 12 + monthly_refund_data['月']
X = monthly_refund_data[['时间']].values  # 自变量：时间（月度）
sales = monthly_refund_data['total_sales'].values  # 因变量：销售额

# 拟合线性回归模型
sales_model = LinearRegression()
sales_model.fit(X, sales)

# 预测未来12个月的销售额
future_months = np.arange(X[-1, 0] + 1, X[-1, 0] + 13).reshape(-1, 1)
predicted_sales = sales_model.predict(future_months)

# 6. 将“时间”转换回年-月的格式
monthly_refund_data['年月'] = monthly_refund_data.apply(lambda row: f"{row['年']}年{row['月']}月", axis=1)

# 可视化退款率回归曲线
plt.figure(figsize=(12, 7))

# 绘制历史销售额的折线图
plt.plot(monthly_refund_data['年月'], sales, color='blue', marker='o', label='历史销售额')

# 绘制销售额回归曲线
plt.plot(monthly_refund_data['年月'], sales_model.predict(X), color='red', label='销售额回归曲线')

# 预测数据（未来12个月）
future_year_month = [f"{(X[-1, 0] + i) // 12}年{(X[-1, 0] + i) % 12}月" for i in range(1, 13)]
plt.plot(future_year_month, predicted_sales, color='green', linestyle='--', label='预测销售额')

# 设置图形的中文标签
plt.xlabel('时间（月）', fontproperties=font_prop)
plt.ylabel('销售额 (元)', fontproperties=font_prop)
plt.title('月度销售额回归分析与预测', fontproperties=font_prop)

# 自动调整横轴标签显示频率，避免重叠
plt.xticks(rotation=45)  # 旋转横轴标签，以免重叠

# 控制横轴标签的显示频率
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=10))

# 显示图形
plt.legend(prop=font_prop)
plt.tight_layout()
plt.show()

# 7. 输出未来12个月的预测销售额
# 将时间转换回年和月
last_time = monthly_refund_data['时间'].max()
predicted_year_month = [(time // 12, time % 12 if time % 12 != 0 else 12) for time in future_months.flatten()]
predicted_year_month = [(year if month != 12 else year + 1, month if month != 12 else 12) for year, month in predicted_year_month]

print("\n未来12个月的预测销售额：")
for i, (year, month) in enumerate(predicted_year_month):
    print(f"预测 {year}年{month}月的销售额为：{predicted_sales[i]:.2f} 元")
