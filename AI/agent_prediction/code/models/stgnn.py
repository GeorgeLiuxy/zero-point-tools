# stgnn.py

import math

import torch
import torch.nn as nn

from layers import TemporalTransformerLayer, SpatialGCNAttentionLayer


class STGNN(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim, out_dim, num_heads=4, num_layers=2, seq_length=8, pred_length=12, device='cpu'):
        super(STGNN, self).__init__()
        """
        引入 GCN 与注意力机制结合的时空图神经网络模型。

        参数：
        - node_feature_dim：节点特征维度。
        - edge_feature_dim：边特征维度。
        - hidden_dim：隐藏层维度。
        - out_dim：输出维度（例如，动作维度）。
        - num_heads：注意力头的数量。
        - num_layers：层数。
        - seq_length：输入序列长度。
        - pred_length：预测序列长度。
        - device：设备（'cpu' 或 'cuda'）。
        """
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.device = device

        # 节点特征编码器
        self.position_encoder = nn.Linear(2, hidden_dim)
        self.goal_encoder = nn.Linear(2, hidden_dim)

        # 历史轨迹序列编码器（使用 Transformer）
        self.history_encoder_layer = nn.TransformerEncoderLayer(d_model=node_feature_dim, nhead=num_heads, batch_first=True)
        self.history_encoder = nn.TransformerEncoder(self.history_encoder_layer, num_layers=num_layers)

        # 局部地图编码器（使用 CNN）
        self.local_map_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )

        # 时间 Transformer 层
        self.temporal_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.temporal_layers.append(TemporalTransformerLayer(input_dim=hidden_dim, num_heads=num_heads))

        # 空间 GCN 注意力层
        self.spatial_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.spatial_layers.append(SpatialGCNAttentionLayer(node_feature_dim=hidden_dim, edge_feature_dim=edge_feature_dim, num_heads=num_heads))

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, out_dim)  # 预测动作（速度、转向角）

    def positional_encoding(self, length, d_model, device):
        pe = torch.zeros(length, d_model, device=device)
        position = torch.arange(0, length, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (length, d_model)

    def forward(self, node_features, adjacency_matrices, edge_features, mask=None):
        """
        前向传播。

        参数：
        - node_features：节点特征列表，每个元素包含 'positions'、'local_maps'、'history_sequences'、'goal_positions'。
        - adjacency_matrices：邻接矩阵序列，形状 (batch_size, seq_length, num_agents, num_agents)。
        - edge_features：边特征张量，形状 (batch_size, seq_length, num_agents, num_agents, edge_feature_dim)。
        - mask：掩码矩阵，形状 (batch_size, seq_length, num_agents)，标记缺失的信息。

        返回：
        - 预测的未来动作，形状 (batch_size, pred_length, num_agents, out_dim)。
        """
        batch_size = len(node_features)
        seq_length = self.seq_length
        num_agents = len(node_features[0]['positions'])
        device = self.device

        # 收集每个时间步的节点特征
        h = []
        for t in range(seq_length):
            positions = torch.tensor(node_features[t]['positions'], dtype=torch.float32).to(device)  # (batch_size, num_agents, 2)
            goal_positions = torch.tensor(node_features[t]['goal_positions'], dtype=torch.float32).to(device)  # (batch_size, num_agents, 2)
            rel_goal_pos = goal_positions - positions  # 相对目标位置

            # 编码位置和目标
            position_feat = self.position_encoder(positions)  # (batch_size, num_agents, hidden_dim)
            goal_feat = self.goal_encoder(rel_goal_pos)  # (batch_size, num_agents, hidden_dim)

            # 编码历史轨迹序列
            history_sequences = node_features[t]['history_sequences']  # 列表
            history_tensors = []
            for agent_id in range(num_agents):
                history_seq = history_sequences[agent_id]
                if history_seq is not None and len(history_seq) > 0:
                    history_seq = torch.tensor(history_seq, dtype=torch.float32).to(device)  # (history_length, feature_dim)
                    # 添加位置编码
                    pe = self.positional_encoding(history_seq.size(0), self.node_feature_dim, device)
                    history_seq = history_seq + pe
                    history_seq = history_seq.unsqueeze(0)  # (1, history_length, feature_dim)
                    history_tensor = self.history_encoder(history_seq)  # (1, history_length, feature_dim)
                    history_tensor = history_tensor.mean(dim=1).squeeze(0)  # (feature_dim)
                else:
                    history_tensor = torch.zeros(self.node_feature_dim).to(device)
                history_tensors.append(history_tensor)
            history_tensors = torch.stack(history_tensors, dim=0)  # (num_agents, node_feature_dim)

            # 编码局部地图
            local_maps = node_features[t]['local_maps']  # 列表
            local_map_tensors = []
            for agent_id in range(num_agents):
                local_map = node_features[t]['local_maps'][agent_id]
                local_map = torch.tensor(local_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, map_size, map_size)
                local_map_feat = self.local_map_encoder(local_map)  # (batch_size, hidden_dim)
                local_map_tensors.append(local_map_feat)
            local_map_tensors = torch.stack(local_map_tensors, dim=0)  # (num_agents, hidden_dim)

            # 合并特征
            features = position_feat + goal_feat + history_tensors + local_map_tensors  # (batch_size, num_agents, hidden_dim)
            h.append(features)
        h = torch.stack(h, dim=1)  # (batch_size, seq_length, num_agents, hidden_dim)

        # 时间 Transformer 层
        for temporal_layer in self.temporal_layers:
            h = temporal_layer(h, src_mask=mask)  # h: (batch_size, seq_length, num_agents, hidden_dim)

        # 对每个时间步，应用空间 GCN 注意力层
        for t in range(seq_length):
            h_t = h[:, t, :, :]  # (batch_size, num_agents, hidden_dim)
            adj_t = adjacency_matrices[:, t, :, :].to(device)  # (batch_size, num_agents, num_agents)
            edge_feat_t = edge_features[:, t, :, :, :].to(device)  # (batch_size, num_agents, num_agents, edge_feature_dim)
            comm_mask_t = adj_t == 0  # True 表示需要掩盖的位置

            for spatial_layer in self.spatial_layers:
                h_t = spatial_layer(h_t, adj_t, edge_feat_t, src_mask=comm_mask_t)  # (batch_size, num_agents, hidden_dim)

            h[:, t, :, :] = h_t  # 更新 h

        # 使用最后的隐藏状态预测未来的动作
        h_pred = h[:, -1, :, :]  # (batch_size, num_agents, hidden_dim)

        pred_actions = []
        for _ in range(self.pred_length):
            action = self.output_layer(h_pred)  # (batch_size, num_agents, out_dim)
            pred_actions.append(action)
            # 可选：如果使用自回归模型，可以更新 h_pred
            # h_pred = update_function(h_pred, action)

        pred_actions = torch.stack(pred_actions, dim=1)  # (batch_size, pred_length, num_agents, out_dim)

        return pred_actions
