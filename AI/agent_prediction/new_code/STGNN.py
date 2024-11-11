import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class STGNN(nn.Module):
    def __init__(self, num_robots, num_features, hidden_dim=64):
        super(STGNN, self).__init__()

        # 定义图卷积层（GCN）
        self.conv1 = GCNConv(num_features, hidden_dim)  # 第一层图卷积
        self.conv2 = GCNConv(hidden_dim, hidden_dim)    # 第二层图卷积
        self.fc = nn.Linear(hidden_dim, num_robots)     # 输出层，可以调整输出维度

        # 机器人数量和特征维度
        self.num_robots = num_robots
        self.num_features = num_features

    def forward(self, x, edge_index):
        """
        前向传播函数

        :param x: 节点特征（机器人的特征，比如位置、速度等）
        :param edge_index: 邻接矩阵（图的边）
        :return: 经过GNN处理后的通信特征
        """
        x = F.relu(self.conv1(x, edge_index))  # 第一层图卷积
        x = F.relu(self.conv2(x, edge_index))  # 第二层图卷积
        x = self.fc(x)  # 输出层
        return x

    def process_communication_network(self, robot_positions, robot_old_positions, adjacency_matrix):
        """
        处理通信网络的函数

        :param robot_positions: 当前机器人的位置
        :param robot_old_positions: 机器人的历史位置
        :param adjacency_matrix: 机器人间的邻接矩阵
        :return: 处理后的通信特征
        """
        # 使用PyTorch Geometric准备图数据
        edge_index = torch.tensor(adjacency_matrix.nonzero(), dtype=torch.long)  # 从邻接矩阵获取边
        node_features = torch.tensor(robot_positions, dtype=torch.float)  # 机器人的位置作为节点特征

        # 将图数据传入GNN
        return self.forward(node_features, edge_index)
