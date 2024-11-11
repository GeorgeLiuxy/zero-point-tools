# layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TemporalTransformerLayer(nn.Module):
    def __init__(self, input_dim, num_heads=4, dropout=0.1):
        super(TemporalTransformerLayer, self).__init__()
        """
        基于 Transformer 的时间注意力层。

        参数：
        - input_dim：输入特征维度。
        - num_heads：注意力头的数量。
        - dropout：Dropout 概率。
        """
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        """
        前向传播。

        参数：
        - x：输入序列，形状 (batch_size, seq_length, num_agents, input_dim)。
        - src_mask：掩码矩阵，形状 (batch_size, seq_length)，标记缺失的数据。

        返回：
        - 输出序列，形状与输入 x 相同。
        """
        batch_size, seq_length, num_agents, input_dim = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch_size, num_agents, seq_length, input_dim)
        x = x.view(batch_size * num_agents, seq_length, input_dim)  # (batch_size * num_agents, seq_length, input_dim)

        # 计算注意力
        attn_output, _ = self.attention(x, x, x, key_padding_mask=src_mask)  # attn_output: (batch_size * num_agents, seq_length, input_dim)
        x = self.layer_norm(x + self.dropout(attn_output))

        # 恢复形状
        x = x.view(batch_size, num_agents, seq_length, input_dim)
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_length, num_agents, input_dim)
        return x

class SpatialGCNAttentionLayer(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, num_heads=4, dropout=0.1):
        super(SpatialGCNAttentionLayer, self).__init__()
        """
        结合 GCN 和注意力机制的空间注意力层。

        参数：
        - node_feature_dim：节点特征维度。
        - edge_feature_dim：边特征维度。
        - num_heads：注意力头的数量。
        - dropout：Dropout 概率。
        """
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.num_heads = num_heads

        # GCN 层
        self.gcn = GCNLayer(node_feature_dim, node_feature_dim)

        # 注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=node_feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(node_feature_dim)
        self.dropout = nn.Dropout(dropout)

        # 边特征映射到节点特征维度
        self.edge_embedding = nn.Linear(edge_feature_dim, node_feature_dim)

    def forward(self, h, adj, edge_features, src_mask=None):
        """
        前向传播。

        参数：
        - h：节点特征，形状 (batch_size, num_agents, node_feature_dim)。
        - adj：邻接矩阵，形状 (batch_size, num_agents, num_agents)。
        - edge_features：边特征，形状 (batch_size, num_agents, num_agents, edge_feature_dim)。
        - src_mask：通信掩码，形状 (batch_size, num_agents, num_agents)，标记由于通信受限而缺失的边。

        返回：
        - 更新后的节点特征，形状 (batch_size, num_agents, node_feature_dim)。
        """
        batch_size, num_agents, node_feature_dim = h.size()

        # GCN 部分
        h_gcn = self.gcn(h, adj)  # (batch_size, num_agents, node_feature_dim)

        # 将边特征映射到节点特征维度
        edge_emb = self.edge_embedding(edge_features)  # (batch_size, num_agents, num_agents, node_feature_dim)

        # 将节点特征与边特征相加
        h_i = h.unsqueeze(2).repeat(1, 1, num_agents, 1)  # (batch_size, num_agents, num_agents, node_feature_dim)
        h_j = h.unsqueeze(1).repeat(1, num_agents, 1, 1)  # (batch_size, num_agents, num_agents, node_feature_dim)
        h_ij = h_j + edge_emb  # (batch_size, num_agents, num_agents, node_feature_dim)

        # 将邻接矩阵和通信掩码结合
        if src_mask is not None:
            combined_mask = adj * src_mask  # (batch_size, num_agents, num_agents)
        else:
            combined_mask = adj  # (batch_size, num_agents, num_agents)

        # 构建注意力掩码
        attn_mask = combined_mask == 0  # True 表示需要掩盖的位置

        # 将节点特征作为查询，邻居特征作为键和值
        h_query = h.view(batch_size * num_agents, 1, node_feature_dim)  # (batch_size * num_agents, 1, node_feature_dim)
        h_key_value = h_ij.view(batch_size * num_agents, num_agents, node_feature_dim)  # (batch_size * num_agents, num_agents, node_feature_dim)
        attn_mask = attn_mask.view(batch_size * num_agents, num_agents)  # (batch_size * num_agents, num_agents)

        attn_output, _ = self.attention(h_query, h_key_value, h_key_value, key_padding_mask=attn_mask)  # attn_output: (batch_size * num_agents, 1, node_feature_dim)
        attn_output = attn_output.squeeze(1)  # (batch_size * num_agents, node_feature_dim)
        h_query = h_query.squeeze(1)  # (batch_size * num_agents, node_feature_dim)
        h_attn = self.layer_norm(h_query + self.dropout(attn_output))
        h_attn = h_attn.view(batch_size, num_agents, node_feature_dim)  # (batch_size, num_agents, node_feature_dim)

        # 将 GCN 和注意力的结果相加
        h_out = h_gcn + h_attn  # (batch_size, num_agents, node_feature_dim)

        return h_out

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        """
        简单的 GCN 层。

        参数：
        - in_features：输入特征维度。
        - out_features：输出特征维度。
        """
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, h, adj):
        """
        前向传播。

        参数：
        - h：节点特征，形状 (batch_size, num_agents, in_features)。
        - adj：邻接矩阵，形状 (batch_size, num_agents, num_agents)。

        返回：
        - 更新后的节点特征，形状 (batch_size, num_agents, out_features)。
        """
        batch_size, num_agents, _ = h.size()
        adj = adj + torch.eye(num_agents, device=h.device).unsqueeze(0)  # 添加自环 (batch_size, num_agents, num_agents)
        degree_matrix = adj.sum(-1).unsqueeze(-1)  # (batch_size, num_agents, 1)
        h = torch.bmm(adj, h) / degree_matrix  # (batch_size, num_agents, in_features)
        h = self.linear(h)  # (batch_size, num_agents, out_features)
        return h
