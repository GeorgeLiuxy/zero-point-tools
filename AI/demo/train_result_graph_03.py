import logging
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置管理
CONFIG = {
    'num_agents': 10,
    'obs_dim': 16,
    'hidden_dim': 32,
    'num_actions': 2,
    'target_radius': 0.1,
    'collision_radius': 0.2,
    'num_epochs': 100,
    'learning_rate': 1e-3,
    'batch_size': 4
}

# 定义强化学习代理模型
class RLAgent(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(RLAgent, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# TemporalBlock 和 TemporalConvNet 定义
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = nn.ConstantPad1d((0, -padding), 0)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(embed_dim=out_channels, num_heads=4)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = nn.ConstantPad1d((0, -padding), 0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        out = out.permute(2, 0, 1)
        out, _ = self.attn(out, out, out)
        out = out.permute(1, 2, 0)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# 时空图神经网络模型
class SpatioTemporalNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SpatioTemporalNet, self).__init__()
        self.gat = GATConv(in_channels, hidden_channels, heads=4, dropout=0.6)
        self.tcn = TemporalConvNet(num_inputs=hidden_channels * 4, num_channels=[hidden_channels, 32])

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.gat(x, edge_index, edge_attr)
        x = x.unsqueeze(2)
        x = self.tcn(x)
        x = x.view(x.size(0), -1)
        return x

# 定义动态通信限制函数
def dynamic_communication_constraint(data, base_snr=0.5, snr_variability=0.2):
    """
    动态模拟通信限制条件，基于距离或信道特性更新边特征。
    """
    x = data.x[:, :2]  # 获取节点位置
    num_edges = data.edge_attr.size(0)

    # 模拟信噪比随时间波动
    snr = base_snr + torch.randn(num_edges) * snr_variability
    snr = torch.clamp(snr, 0, 1)  # 保证 SNR 在 [0, 1] 范围内
    data.edge_attr[:, 0] = data.edge_attr[:, 0] * snr  # 调整边特征

    return data

# 奖励计算
def calculate_reward(actions, data, target_radius=0.1, collision_radius=0.2):
    distance_to_target = torch.norm(actions - data.y, dim=1)
    reward = -distance_to_target
    reached_target = distance_to_target < target_radius
    reward[reached_target] += 1.0
    num_agents = actions.size(0)
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            if torch.norm(actions[i] - actions[j]) < collision_radius:
                reward[i] -= 1.0
                reward[j] -= 1.0
    return reward

# 强化学习训练
def train_rl(agent, network, data_loader, optimizer, num_epochs=200):
    loss_list = []
    snr_list = []
    prediction_error_list = []

    for epoch in range(num_epochs):
        total_loss = 0
        avg_snr = 0  # 平均SNR记录
        total_prediction_error = 0  # 总预测误差记录
        num_batches = len(data_loader)

        for batch in data_loader:
            # 检查 batch 是否是 list，如果是，则取其中的第一个元素
            if isinstance(batch, list):
                batch = batch[0]

            # 动态调整通信限制
            batch = dynamic_communication_constraint(batch)
            avg_snr += batch.edge_attr[:, 0].mean().item()

            optimizer.zero_grad()
            spatio_temporal_output = network(batch)
            actions = agent(spatio_temporal_output)
            reward = calculate_reward(actions, batch)
            loss = -torch.mean(reward)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 计算并记录行为预测误差
            prediction_error = torch.norm(actions - batch.y, dim=1).mean().item()
            total_prediction_error += prediction_error

        # 记录每个epoch的数据
        avg_loss = total_loss / num_batches
        avg_snr /= num_batches
        avg_prediction_error = total_prediction_error / num_batches

        loss_list.append(avg_loss)
        snr_list.append(avg_snr)
        prediction_error_list.append(avg_prediction_error)

        # 输出每个智能体的预测行为和目标行为
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}, Avg SNR: {avg_snr:.4f}, Avg Prediction Error: {avg_prediction_error:.4f}")

    # 绘制图表
    plot_training_progress(loss_list, snr_list, prediction_error_list)


# 绘制训练进展图表
def plot_training_progress(loss_list, snr_list, prediction_error_list):
    epochs = range(1, len(loss_list) + 1)

    plt.figure(figsize=(12, 8))

    # 绘制损失
    plt.subplot(3, 1, 1)
    plt.plot(epochs, loss_list, label='Loss', color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')

    # 绘制SNR
    plt.subplot(3, 1, 2)
    plt.plot(epochs, snr_list, label='Average SNR', color='g')
    plt.xlabel('Epochs')
    plt.ylabel('SNR')
    plt.title('Average SNR Over Epochs')

    # 绘制预测误差
    plt.subplot(3, 1, 3)
    plt.plot(epochs, prediction_error_list, label='Prediction Error', color='r')
    plt.xlabel('Epochs')
    plt.ylabel('Prediction Error')
    plt.title('Average Prediction Error Over Epochs')

    plt.tight_layout()
    plt.show()

# SCA路径规划
def sca_path_planning(data, num_iterations=50, initial_temp=10.0, cooling_rate=0.9, move_scale=0.05):
    current_pos = data.x[:, :2]
    target_pos = data.y
    path = [current_pos.clone()]  # 记录每个迭代的路径
    temp = initial_temp
    for _ in range(num_iterations):
        # 控制随机移动的步长
        neighbor_paths = current_pos + (torch.randn_like(current_pos) * move_scale)
        neighbor_cost = torch.norm(neighbor_paths - target_pos, dim=1).sum()
        delta_cost = neighbor_cost - torch.norm(current_pos - target_pos, dim=1).sum()
        if delta_cost < 0 or torch.rand(1).item() < torch.exp(-delta_cost / temp):
            current_pos = neighbor_paths
        path.append(current_pos.clone())
        temp *= cooling_rate
    return path


# 路径可视化函数
# 路径可视化函数
def plot_paths(rl_path, sca_path, target_pos, obstacles=None, observation_radius=0.5):
    plt.figure(figsize=(10, 8))

    # 将路径堆叠成 numpy 数组，方便绘图
    rl_path = torch.stack(rl_path).numpy() if isinstance(rl_path, list) else rl_path.detach().cpu().numpy()
    sca_path = torch.stack(sca_path).numpy() if isinstance(sca_path, list) else sca_path.detach().cpu().numpy()
    num_steps = rl_path.shape[0]  # 使用 RL 路径的步数作为渐变色的参考

    # 使用 colormap 获取渐变颜色
    cmap = plt.get_cmap('viridis')

    # 绘制障碍物
    if obstacles is not None:
        for obs in obstacles:
            plt.gca().add_patch(plt.Rectangle((obs[0] - 0.1, obs[1] - 0.1), 0.2, 0.2, color='black'))

    # 为每个智能体绘制路径
    for agent_idx in range(rl_path.shape[1]):
        # 绘制观测范围
        for point in rl_path[:, agent_idx]:
            # 确保 point 是一个 (x, y) 坐标对
            if point.shape == (2,):
                circle = plt.Circle((point[0], point[1]), observation_radius, color='gray', alpha=0.1)
                plt.gca().add_patch(circle)

        # 绘制 RL 路径，使用粗线条和蓝色
        for step in range(num_steps - 1):
            color = cmap(step / num_steps)
            plt.plot(
                [rl_path[step, agent_idx, 0], rl_path[step + 1, agent_idx, 0]],
                [rl_path[step, agent_idx, 1], rl_path[step + 1, agent_idx, 1]],
                color='blue', linewidth=3, label="RL Path" if step == 0 and agent_idx == 0 else ""
            )
            plt.scatter(rl_path[step, agent_idx, 0], rl_path[step, agent_idx, 1], color='blue', s=20)

        # 绘制 SCA 路径，使用粗线条和绿色
        for step in range(len(sca_path) - 1):
            plt.plot(
                [sca_path[step, agent_idx, 0], sca_path[step + 1, agent_idx, 0]],
                [sca_path[step, agent_idx, 1], sca_path[step + 1, agent_idx, 1]],
                color='green', linewidth=3, linestyle='--', label="SCA Path" if step == 0 and agent_idx == 0 else ""
            )
            plt.scatter(sca_path[step, agent_idx, 0], sca_path[step, agent_idx, 1], color='green', s=20)

        # 在路径的最后一个点和目标点之间绘制连接线
        plt.plot(
            [rl_path[-1, agent_idx, 0], target_pos[agent_idx, 0].item()],
            [rl_path[-1, agent_idx, 1], target_pos[agent_idx, 1].item()],
            'k--', linewidth=1.5
        )

        plt.plot(
            [sca_path[-1, agent_idx, 0], target_pos[agent_idx, 0].item()],
            [sca_path[-1, agent_idx, 1], target_pos[agent_idx, 1].item()],
            'g--', linewidth=1
        )

        # 绘制目标点
        plt.scatter(target_pos[agent_idx, 0].item(), target_pos[agent_idx, 1].item(), color='red', marker='x', s=100)

    # 添加图例和标题
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.title("RL vs SCA Path Comparison with Obstacles and Target")
    plt.show()


# 计算路径的总步数和总长度
def calculate_path_metrics(path, target):
    total_steps = len(path)
    total_length = sum(torch.norm(path[i] - path[i + 1]) for i in range(total_steps - 1)).item()

    # 计算每个智能体最终位置到目标位置的距离
    final_distances = torch.norm(path[-1] - target, dim=1)
    avg_final_distance = final_distances.mean().item()  # 计算平均值作为指标

    return total_steps, total_length, avg_final_distance


# 数据生成函数
def generate_data_for_agent(agent_id, num_nodes=10, obs_dim=16, edge_attr_dim=4, history_len=5, num_obstacles=20, safe_distance=0.3):
    # 根据 agent_id 设置智能体的初始位置
    initial_x = torch.tensor([agent_id * 0.5, 0.0])  # 基于 agent_id 设置偏移
    position = torch.randn((num_nodes, 2)) + initial_x  # 偏移位置坐标
    features = torch.randn((num_nodes, obs_dim - 2))  # 其他特征
    x = torch.cat((position, features), dim=1)  # 合并

    # x = torch.randn((num_nodes, obs_dim))
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
    edge_attr = torch.randn((edge_index.size(1), edge_attr_dim))
    snr = torch.rand(edge_attr.size(0))
    edge_attr[:, 0] = edge_attr[:, 0] * snr
    history = torch.randn((num_nodes, history_len, obs_dim))

    # 随机生成障碍物的位置
    obstacles = []
    for _ in range(num_obstacles):
        obs_x = random.uniform(-2, 2)
        obs_y = random.uniform(-2, 2)
        obstacles.append([obs_x, obs_y])
    obstacles = torch.tensor(obstacles)  # 转换为 tensor 格式

    # 生成目标点，确保目标点不会与障碍物重叠
    target_points = []
    for _ in range(num_nodes):
        while True:
            # 随机生成目标点
            target_x = random.uniform(-2, 2)
            target_y = random.uniform(-2, 2)
            target_point = torch.tensor([target_x, target_y])

            # 计算目标点与所有障碍物的距离
            distances = torch.norm(obstacles - target_point, dim=1)

            # 检查目标点是否与障碍物保持安全距离
            if torch.all(distances > safe_distance):
                target_points.append(target_point)
                break  # 退出循环，找到符合条件的目标点

    # 转换为 tensor，方便后续使用
    y = torch.stack(target_points)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, history=history), obstacles


# 主流程
def main():
    num_agents = CONFIG['num_agents']
    obs_dim = CONFIG['obs_dim']
    hidden_dim = CONFIG['hidden_dim']
    num_actions = CONFIG['num_actions']
    num_epochs = CONFIG['num_epochs']
    learning_rate = CONFIG['learning_rate']
    batch_size = CONFIG['batch_size']

    # 生成初始数据集
    data_list = [generate_data_for_agent(i, num_nodes=num_agents, obs_dim=obs_dim) for i in range(num_agents)]
    data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

    network = SpatioTemporalNet(in_channels=obs_dim, hidden_channels=hidden_dim, out_channels=num_actions)
    agent = RLAgent(input_dim=hidden_dim, action_dim=num_actions)
    optimizer = optim.Adam(list(network.parameters()) + list(agent.parameters()), lr=learning_rate)

    # 训练
    train_rl(agent, network, data_loader, optimizer, num_epochs=num_epochs)

    # 路径对比
    for data, obstacles in data_list:
        # 传统SCA路径规划
        sca_path = sca_path_planning(data)
        # 生成 RL 路径
        sca_steps, sca_length, sca_final_distance = calculate_path_metrics(sca_path, data.y)

        # 使用RL模型的路径规划
        with torch.no_grad():
            rl_output = network(data)
            rl_path = agent(rl_output)
            rl_steps, rl_length, rl_final_distance = calculate_path_metrics(rl_path, data.y)

        # 确保 rl_path 和 sca_path 是三维张量
        if isinstance(rl_path, list):
            rl_path = torch.stack(rl_path)  # 将列表转换为张量
        if isinstance(sca_path, list):
            sca_path = torch.stack(sca_path)  # 将列表转换为张量

        # 检查并调整形状为 (num_steps, num_agents, 2)
        if rl_path.dim() == 2:  # 如果是二维 (num_steps, 2)，添加额外维度
            rl_path = rl_path.unsqueeze(1)  # 变成 (num_steps, 1, 2)
        if sca_path.dim() == 2:  # 同样处理 sca_path
            sca_path = sca_path.unsqueeze(1)

        plot_paths(rl_path, sca_path, target_pos=data.y, obstacles=obstacles)

        # 输出对比结果
        logger.info(f"Agent Comparison:")
        logger.info(f" - SCA Path: Steps={sca_steps}, Length={sca_length:.4f}, Final Distance to Target={sca_final_distance:.4f}")
        logger.info(f" - RL Path: Steps={rl_steps}, Length={rl_length:.4f}, Final Distance to Target={rl_final_distance:.4f}")


if __name__ == "__main__":
    main()
