import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import logging
import matplotlib.pyplot as plt

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

# SCA路径规划函数
def sca_path_planning(data, num_iterations=100, initial_temp=10.0, cooling_rate=0.95):
    current_pos = data.x[:, :2]
    target_pos = data.y
    path = [current_pos.clone()]  # 记录路径
    temp = initial_temp
    for _ in range(num_iterations):
        neighbor_paths = current_pos + torch.randn_like(current_pos) * 0.1
        neighbor_cost = torch.norm(neighbor_paths - target_pos, dim=1).sum()
        delta_cost = neighbor_cost - torch.norm(current_pos - target_pos, dim=1).sum()
        if delta_cost < 0 or torch.rand(1).item() < torch.exp(-delta_cost / temp):
            current_pos = neighbor_paths
        path.append(current_pos.clone())
        temp *= cooling_rate
    return path  # 返回每次迭代的路径

# 新增：路径可视化函数
def plot_paths(paths, target_pos):
    plt.figure(figsize=(8, 6))
    paths = torch.stack(paths).numpy()  # 转换为numpy以便绘图
    for agent_idx in range(paths.shape[1]):
        plt.plot(paths[:, agent_idx, 0], paths[:, agent_idx, 1], marker='o', label=f'Agent {agent_idx + 1}')
        plt.scatter(target_pos[agent_idx, 0], target_pos[agent_idx, 1], color='red', marker='x', label=f'Target {agent_idx + 1}' if agent_idx == 0 else None)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.title("Path Planning for Agents")
    plt.show()

# 新增：训练函数 train_rl
def train_rl(agent, network, data_loader, optimizer, num_epochs=100):
    loss_list = []
    snr_list = []
    prediction_error_list = []

    for epoch in range(num_epochs):
        total_loss = 0
        avg_snr = 0  # 平均SNR记录
        total_prediction_error = 0  # 总预测误差记录
        num_batches = len(data_loader)

        for batch in data_loader:
            # 动态调整通信限制
            batch = dynamic_communication_constraint(batch)
            avg_snr += batch.edge_attr[:, 0].mean().item()

            optimizer.zero_grad()
            spatio_temporal_output = network(batch)
            actions = agent(spatio_temporal_output)

            # 使用示例损失（或替换为具体的损失计算）
            loss = torch.mean(actions)
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

# 数据生成函数
def generate_data_for_agent(agent_id, num_nodes=10, obs_dim=16, edge_attr_dim=4, history_len=5):
    x = torch.randn((num_nodes, obs_dim))
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
    edge_attr = torch.randn((edge_index.size(1), edge_attr_dim))
    snr = torch.rand(edge_attr.size(0))
    edge_attr[:, 0] = edge_attr[:, 0] * snr
    history = torch.randn((num_nodes, history_len, obs_dim))
    y = torch.randn((num_nodes, 2))
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, history=history)

# 主流程
def main():
    num_agents = CONFIG['num_agents']
    obs_dim = CONFIG['obs_dim']
    hidden_dim = CONFIG['hidden_dim']
    num_actions = CONFIG['num_actions']
    num_epochs = CONFIG['num_epochs']
    learning_rate = CONFIG['learning_rate']
    batch_size = CONFIG['batch_size']

    # 生成初始数据
    data_list = [generate_data_for_agent(i, num_nodes=num_agents, obs_dim=obs_dim) for i in range(num_agents)]
    data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

    network = SpatioTemporalNet(in_channels=obs_dim, hidden_channels=hidden_dim, out_channels=num_actions)
    agent = RLAgent(input_dim=hidden_dim, action_dim=num_actions)
    optimizer = optim.Adam(list(network.parameters()) + list(agent.parameters()), lr=learning_rate)

    # 训练
    train_rl(agent, network, data_loader, optimizer, num_epochs=num_epochs)

    # 路径规划并可视化
    for data in data_list:
        paths = sca_path_planning(data)
        plot_paths(paths, data.y)


if __name__ == "__main__":
    main()
