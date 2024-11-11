import numpy as np

# 假设参数
grid_size = 100
grid_resolution = 1
totalRobot = 8
totalDynamic = 5
lambda_wavelength = 1  # 信号波长（假设值）
Gr = Gt = 14  # 接收和发射天线增益
Pt = 1  # 发射功率1w
Pn = 0.01  # 噪声功率0.01w
B = 10e6  # 带宽
beta = 30  # 莱斯因子
snr_threshold = 10 # SNR阈值

def ricean_fading(beta):
    """生成莱斯衰落系数"""
    h_LoS = 1  # 假设视距分量幅度为1
    h_NLoS = np.sqrt(1/2) * np.random.randn() + 1j * np.sqrt(1/2) * np.random.randn()    # 瑞利衰落分量
    return np.sqrt(beta / (beta + 1)) * h_LoS + np.sqrt(1 / (beta + 1)) * h_NLoS


def large_scale_fading(d):
    """计算大尺度衰落的路径损耗因子"""
    return Gr * Gt * ((lambda_wavelength / (4 * np.pi * d)) ** 2)


def calculate_snr(d, Pt = 1, Pn = 0.01, beta = 30, los=True):
    """计算信噪比SNR"""
    if los:
        h = ricean_fading(beta)
    else:
        h = np.sqrt(1/2) * np.random.randn() + 1j * np.sqrt(1/2) * np.random.randn()  # 瑞利衰落

    path_gain = large_scale_fading(d)
    h_total = np.sqrt(path_gain) * h
    snr = np.abs(h_total) * (Pt / Pn)
    # print(snr)
    return snr


def is_link_active(d, count, snr_threshold=10):
    """判断通信链路是否活跃"""
    los = np.random.rand() < count  # 假设count%的概率是LoS
    snr = calculate_snr(d, Pt, Pn, beta, los)
    snr = max(snr, 1e-10)  # 确保snr不为0，避免log10(0)的情况
    capacity = calculate_capacity(B, snr)
    return (10 * np.log10(snr)) > snr_threshold, capacity, los, snr



def calculate_capacity(B, snr):
    """计算信道容量"""
    return B * np.log2(1 + snr)


# # 主循环（简化表示，省略了栅格地图和障碍物生成等部分）
# # 假设robotNew已经包含了所有机器人的当前位置
# point1 = np.array([12, 12])
# point2 = np.array([25, 25])
#
# d = np.linalg.norm(point1 - point2)
# # 假设距离d的值，测试通信是否连接
# d = 12
#
# # 假设有一个函数check_los来检测LoS/NLoS条件与莱斯因子变化（这里用简化的逻辑代替）
#
# # los = np.random.rand() < 0.9  # 假设90%的概率是LoS
# # snr = calculate_snr(d, Pt, Pn, beta, los)
# flag,capacity,los,snr = is_link_active(d,snr_threshold = 10)
# if flag:
#     capacity = calculate_capacity(B, snr)
#     print(f"Robot {1} to Robot {2}: SNR={snr:.2f}, Capacity={capacity:.2f} bps, LoS={los}")
# else:
#     print(f"Robot {1} to Robot {2}: Link is inactive due to low SNR")

