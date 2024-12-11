import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy import stats
import platform
import sys

# 1. 设置中文字体
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

# 2. 加载和预处理数据
def load_and_prepare_data(file_path):
    """
    加载Excel数据文件，计算退款率，并删除买家实际支付金额为0的无效数据。

    参数:
        file_path (str): Excel文件的路径。

    返回:
        X (numpy.ndarray): 自变量（总金额）的二维数组。
        y (numpy.ndarray): 因变量（退款率）的数组。
        data (pandas.DataFrame): 处理后的数据框。
    """
    # 加载数据
    try:
        data = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"文件未找到，请检查路径是否正确: {file_path}")
        sys.exit(1)

    # 检查必要的列是否存在
    required_columns = ['订单编号', '总金额', '买家实际支付金额', '退款金额']
    for col in required_columns:
        if col not in data.columns:
            print(f"数据中缺少必要的列: {col}")
            sys.exit(1)

    # 显示前几行数据以进行检查
    print("原始数据预览：")
    print(data.head())

    # 计算退款率
    data['退款率'] = data['退款金额'] / data['买家实际支付金额']

    # 删除买家实际支付金额为0的行
    data = data[data['买家实际支付金额'] != 0]

    # 删除可能存在的无效退款率（如负值或超过1的值）
    data = data[(data['退款率'] >= 0) & (data['退款率'] <= 1)]

    # 显示计算后的数据预览
    print("\n计算退款率后的数据预览：")
    print(data.head())

    # 数据准备
    X = data['总金额'].values.reshape(-1, 1)  # 自变量：总金额
    y = data['退款率'].values  # 因变量：退款率

    # 打印X和y以确认
    print("\n自变量（总金额）X：")
    print(X.flatten())
    print("\n因变量（退款率）y：")
    print(y)

    return X, y, data

# 替换为你的Excel文件路径
file_path = 'shanghai.xlsx'  # 请确保文件路径正确
X, y, data = load_and_prepare_data(file_path)

# 检查数据点数量
def check_data_size(X, y):
    """
    检查数据点的数量，确保有足够的数据进行回归分析。

    参数:
        X (numpy.ndarray): 自变量的二维数组。
        y (numpy.ndarray): 因变量的数组。
    """
    n_samples = X.shape[0]
    print(f"\n数据点数量：{n_samples}")
    if n_samples < 2:
        print("数据点太少，无法进行回归分析。")
        sys.exit(1)
    elif n_samples < 10:
        print("注意：数据点较少，模型可能不稳定。")

check_data_size(X, y)

# 3. 创建并训练线性回归模型
def train_linear_regression(X, y):
    """
    创建并训练线性回归模型。

    参数:
        X (numpy.ndarray): 自变量的二维数组。
        y (numpy.ndarray): 因变量的数组。

    返回:
        model (LinearRegression): 训练好的线性回归模型。
        y_pred (numpy.ndarray): 模型对X的预测值。
    """
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred

model, y_pred = train_linear_regression(X, y)

# 4. 评估模型
def evaluate_model(model, y, y_pred):
    """
    计算并打印线性回归模型的评估指标。

    参数:
        model (LinearRegression): 训练好的线性回归模型。
        y (numpy.ndarray): 实际的因变量值。
        y_pred (numpy.ndarray): 预测的因变量值。
    """
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f'\n回归系数（斜率）：{model.coef_[0]:.4f}')
    print(f'截距：{model.intercept_:.4f}')
    print(f'MSE（均方误差）：{mse:.6f}')
    print(f'R²（决定系数）：{r2:.4f}')

evaluate_model(model, y, y_pred)

# 5. 绘制散点图和回归线
def plot_regression(X, y, model):
    """
    绘制实际值的散点图、回归线以及预测点（如果有）。

    参数:
        X (numpy.ndarray): 自变量的二维数组。
        y (numpy.ndarray): 因变量的数组。
        model (LinearRegression): 训练好的线性回归模型。
    """
    plt.figure(figsize=(12, 8))

    # 绘制实际数据的散点图
    plt.scatter(X, y, color='blue', label='实际值', s=100, alpha=0.7, edgecolors='k')

    # 生成更密集的X值用于绘制回归线
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_range_pred = model.predict(X_range)

    # 绘制回归线
    plt.plot(X_range, y_range_pred, color='red', label='回归线', linewidth=2)

    # 绘制预测值与实际值的对比
    plt.scatter(X, y_pred, color='orange', label='预测值', marker='o', s=100, facecolors='none', edgecolors='orange', linewidths=2)

    # 添加新预测点（可选）
    new_X = np.array([[1500], [2500], [3500]])  # 替换为你的新总金额数据
    new_y_pred = model.predict(new_X)
    plt.scatter(new_X, new_y_pred, color='green', label='新预测值', marker='x', s=200, linewidths=2)

    # 图表标题和标签
    plt.title('退款率预测：一元线性回归', fontsize=20)
    plt.xlabel('总金额', fontsize=16)
    plt.ylabel('退款率', fontsize=16)

    # 图例和网格
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.show()

plot_regression(X, y, model)

# 6. 绘制残差图及残差分布
def plot_residuals(X, y, y_pred):
    """
    绘制残差图，以评估模型的拟合效果。

    参数:
        X (numpy.ndarray): 自变量的二维数组。
        y (numpy.ndarray): 实际的因变量值。
        y_pred (numpy.ndarray): 预测的因变量值。
    """
    residuals = y - y_pred  # 计算残差

    # 绘制残差散点图
    plt.figure(figsize=(12, 8))
    plt.scatter(X, residuals, color='purple', s=100, alpha=0.7, edgecolors='k')
    plt.axhline(0, color='black', linestyle='--', linewidth=2)

    plt.title('残差图', fontsize=20)
    plt.xlabel('总金额', fontsize=16)
    plt.ylabel('残差', fontsize=16)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # 绘制残差分布直方图和QQ图
    plt.figure(figsize=(14, 6))

    # 直方图
    plt.subplot(1, 2, 1)
    sns.histplot(residuals, kde=True, color='purple', edgecolor='black')
    plt.title('残差分布直方图', fontsize=16)
    plt.xlabel('残差', fontsize=14)
    plt.ylabel('频数', fontsize=14)

    # QQ图
    plt.subplot(1, 2, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('残差QQ图', fontsize=16)

    plt.tight_layout()
    plt.show()

    # 打印残差统计信息
    print(f'残差均值：{np.mean(residuals):.6f}')
    print(f'残差标准差：{np.std(residuals):.6f}')

plot_residuals(X, y, y_pred)

# 7. 可选：多项式回归（如果线性回归效果不佳）
def train_polynomial_regression(X, y, degree=2):
    """
    创建并训练多项式回归模型。

    参数:
        X (numpy.ndarray): 自变量的二维数组。
        y (numpy.ndarray): 因变量的数组。
        degree (int): 多项式的度数。

    返回:
        model_poly (Pipeline): 训练好的多项式回归模型。
        y_pred_poly (numpy.ndarray): 模型对X的预测值。
    """
    model_poly = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model_poly.fit(X, y)
    y_pred_poly = model_poly.predict(X)
    return model_poly, y_pred_poly

def evaluate_polynomial_model(model_poly, y, y_pred_poly, degree=2):
    """
    计算并打印多项式回归模型的评估指标。

    参数:
        model_poly (Pipeline): 训练好的多项式回归模型。
        y (numpy.ndarray): 实际的因变量值。
        y_pred_poly (numpy.ndarray): 预测的因变量值。
        degree (int): 多项式的度数。
    """
    mse_poly = mean_squared_error(y, y_pred_poly)
    r2_poly = r2_score(y, y_pred_poly)

    # 获取线性回归部分的系数
    linear_model = model_poly.named_steps['linearregression']
    print(f'\n多项式回归（degree={degree}) 截距：{linear_model.intercept_:.4f}')
    print(f'多项式回归（degree={degree}) 回归系数：{linear_model.coef_}')
    print(f'MSE（均方误差）：{mse_poly:.6f}')
    print(f'R²（决定系数）：{r2_poly:.4f}')

def plot_polynomial_regression(X, y, model_poly, degree=2):
    """
    绘制多项式回归的散点图和回归曲线。

    参数:
        X (numpy.ndarray): 自变量的二维数组。
        y (numpy.ndarray): 因变量的数组。
        model_poly (Pipeline): 训练好的多项式回归模型。
        degree (int): 多项式的度数。
    """
    plt.figure(figsize=(12, 8))

    # 绘制实际数据的散点图
    plt.scatter(X, y, color='blue', label='实际值', s=100, alpha=0.7, edgecolors='k')

    # 生成更密集的X值用于绘制回归曲线
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_range_pred_poly = model_poly.predict(X_range)

    # 绘制回归曲线
    plt.plot(X_range, y_range_pred_poly, color='red', label=f'多项式回归 (degree={degree})', linewidth=2)

    # 图表标题和标签
    plt.title(f'退款率预测：多项式回归 (degree={degree})', fontsize=20)
    plt.xlabel('总金额', fontsize=16)
    plt.ylabel('退款率', fontsize=16)

    # 图例和网格
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.show()

# 示例：多项式回归（degree=2）
degree = 2
model_poly, y_pred_poly = train_polynomial_regression(X, y, degree)
evaluate_polynomial_model(model_poly, y, y_pred_poly, degree)
plot_polynomial_regression(X, y, model_poly, degree)

# 8. 总结与建议
def summarize_results(model, y, y_pred, model_poly=None, y_pred_poly=None, degree=2):
    """
    打印模型总结信息。

    参数:
        model (LinearRegression): 线性回归模型。
        y (numpy.ndarray): 实际的因变量值。
        y_pred (numpy.ndarray): 预测的因变量值。
        model_poly (Pipeline, optional): 多项式回归模型。
        y_pred_poly (numpy.ndarray, optional): 多项式回归预测值。
        degree (int, optional): 多项式的度数。
    """
    print("\n=== 模型总结 ===")
    print("线性回归模型：")
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"  截距：{model.intercept_:.4f}")
    print(f"  回归系数（斜率）：{model.coef_[0]:.4f}")
    print(f"  MSE：{mse:.6f}")
    print(f"  R²：{r2:.4f}")

    if model_poly is not None and y_pred_poly is not None:
        mse_poly = mean_squared_error(y, y_pred_poly)
        r2_poly = r2_score(y, y_pred_poly)
        print(f"\n多项式回归（degree={degree}) 模型：")
        linear_model = model_poly.named_steps['linearregression']
        print(f"  截距：{linear_model.intercept_:.4f}")
        print(f"  回归系数：{linear_model.coef_}")
        print(f"  MSE：{mse_poly:.6f}")
        print(f"  R²：{r2_poly:.4f}")

summarize_results(model, y, y_pred, model_poly, y_pred_poly, degree)
