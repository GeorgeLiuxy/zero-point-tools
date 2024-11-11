import numpy as np

def is_link_active(distance, p_los):
    """
    计算两个智能体之间的通信链路。

    参数：
    - distance：两智能体之间的距离。
    - p_los：视距（LOS）概率。

    返回：
    - link_active：链路是否激活（1 或 0）。
    - capacity：信道容量（根据香农公式）。
    - snr：信噪比。
    - p_los：视距概率。
    """
    # 通信参数
    max_distance = 50  # 最大通信距离
    noise_power = 1e-9  # 噪声功率
    transmit_power = 1e-3  # 发射功率

    # 路径损耗模型
    path_loss_exponent = 2.0
    path_loss = (distance ** path_loss_exponent)
    if path_loss == 0:
        path_loss = 1e-9  # 防止除零

    # 信号功率计算
    signal_power = transmit_power / path_loss

    # 信噪比计算
    snr = signal_power / noise_power

    # 信道容量（比特/秒/赫兹）
    capacity = np.log2(1 + snr)

    # 判断链路是否激活
    if distance <= max_distance and p_los > 0.1:
        link_active = 1
    else:
        link_active = 0
        capacity = 0
        snr = 0

    return link_active, capacity, snr, p_los

def simulate_packet_loss(history_sequence, packet_loss_rate):
    """
    模拟通信中信息的丢失。

    参数：
    - history_sequence：历史轨迹序列。
    - packet_loss_rate：数据包丢失率。

    返回：
    - transmitted_sequence：传输后的历史轨迹序列，可能包含缺失。
    """
    if np.random.rand() < packet_loss_rate:
        return None  # 整个序列丢失
    else:
        return history_sequence  # 无丢失

