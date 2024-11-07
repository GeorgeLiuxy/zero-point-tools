import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import logging
from typing import List

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置管理（可用配置文件替代）
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

# 定义强化学习代理模型，用于多智能体行为预测
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
def train_rl(agent, network, data_loader, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in data_loader:
            optimizer.zero_grad()
            spatio_temporal_output = network(batch)
            actions = agent(spatio_temporal_output)
            reward = calculate_reward(actions, batch)
            loss = -torch.mean(reward)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# SCA路径规划
def sca_path_planning(data, num_iterations=100, initial_temp=10.0, cooling_rate=0.95):
    current_pos = data.x[:, :2]
    target_pos = data.y
    best_paths = current_pos.clone()
    best_cost = torch.norm(best_paths - target_pos, dim=1).sum()
    temp = initial_temp
    for _ in range(num_iterations):
        neighbor_paths = best_paths + torch.randn_like(best_paths) * 0.1
        neighbor_cost = torch.norm(neighbor_paths - target_pos, dim=1).sum()
        delta_cost = neighbor_cost - best_cost
        if delta_cost < 0 or torch.rand(1).item() < torch.exp(-delta_cost / temp):
            best_paths = neighbor_paths
            best_cost = neighbor_cost
        temp *= cooling_rate
    return best_paths

# 主流程
def main():
    num_agents = CONFIG['num_agents']
    obs_dim = CONFIG['obs_dim']
    hidden_dim = CONFIG['hidden_dim']
    num_actions = CONFIG['num_actions']
    num_epochs = CONFIG['num_epochs']
    learning_rate = CONFIG['learning_rate']
    batch_size = CONFIG['batch_size']

    data_list = [generate_data_for_agent(i, num_nodes=num_agents, obs_dim=obs_dim) for i in range(num_agents)]
    data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

    network = SpatioTemporalNet(in_channels=obs_dim, hidden_channels=hidden_dim, out_channels=num_actions)
    agent = RLAgent(input_dim=hidden_dim, action_dim=num_actions)
    optimizer = optim.Adam(list(network.parameters()) + list(agent.parameters()), lr=learning_rate)

    train_rl(agent, network, data_loader, optimizer, num_epochs=num_epochs)

    for data in data_list:
        paths = sca_path_planning(data)
        logger.info(f"Planned paths for agent: {paths}")

if __name__ == "__main__":
    main()
