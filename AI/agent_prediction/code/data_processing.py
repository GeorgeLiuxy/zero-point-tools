import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class AgentDataset(Dataset):
    def __init__(self, data_dir, seq_length=8, pred_length=12):
        """
        初始化数据集。

        参数：
        - data_dir：数据文件的目录。
        - seq_length：历史序列长度。
        - pred_length：预测序列长度。
        """
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.data = []

        # 加载所有数据文件
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.pkl'):
                file_path = os.path.join(data_dir, file_name)
                with open(file_path, 'rb') as f:
                    sim_data_list = pickle.load(f)
                    for sim_data in sim_data_list:
                        self._process_simulation(sim_data)

    def _process_simulation(self, sim_data):
        """
        处理单次仿真数据，生成样本。

        参数：
        - sim_data：单次仿真的数据字典。
        """
        timesteps = sim_data['timesteps']
        agent_states = sim_data['agent_states']
        adjacency_matrices = sim_data['adjacency_matrices']
        edge_features = sim_data['edge_features']
        num_agents = sim_data['totalRobot']
        robot_historys = sim_data['robot_historys']  # 新增：每个智能体的历史轨迹序列

        total_timesteps = len(timesteps)
        seq_length = self.seq_length
        pred_length = self.pred_length

        # 滑动窗口生成样本
        for t in range(total_timesteps - seq_length - pred_length + 1):
            input_sequence = {
                'current_positions': [],     # 当前坐标
                'local_maps': [],            # 局部观测栅格图
                'history_sequences': [],     # 历史轨迹序列
                'goal_positions': [],        # 终点坐标
                'adjacency_matrices': [],
                'edge_features': [],
            }
            target_sequence = {
                'actions': [],  # 需要预测的行为：转向角和速度
                'adjacency_matrices': [],
            }
            for i in range(seq_length):
                state = agent_states[t + i]
                input_sequence['current_positions'].append(state['positions'])
                input_sequence['history_sequences'].append(state['history_sequences'])
                input_sequence['local_maps'].append(state['local_maps'])
                input_sequence['goal_positions'].append(state['goal_positions'])
                input_sequence['adjacency_matrices'].append(adjacency_matrices[t + i])
                input_sequence['edge_features'].append(edge_features[t + i])

            for i in range(pred_length):
                state = agent_states[t + seq_length + i]
                # 目标是预测转向角和速度，可以从 state 中获取
                target_sequence['actions'].append(state['actions'])
                target_sequence['adjacency_matrices'].append(adjacency_matrices[t + seq_length + i])

            self.data.append((input_sequence, target_sequence))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本。

        返回：
        - 输入特征、目标值、掩码等。
        """
        input_seq, target_seq = self.data[idx]

        # 转换为张量
        # 节点特征：当前坐标、局部地图、历史轨迹序列、终点坐标
        node_features = []
        for i in range(self.seq_length):
            positions = input_seq['current_positions'][i]          # (num_agents, 2)
            local_maps = input_seq['local_maps'][i]                # (num_agents, map_size, map_size)
            history_sequences = input_seq['history_sequences'][i]  # (num_agents, history_length, feature_dim)
            goal_positions = input_seq['goal_positions'][i]        # (num_agents, 2)
            features = {
                'positions': positions,
                'local_maps': local_maps,
                'history_sequences': history_sequences,
                'goal_positions': goal_positions,
            }
            node_features.append(features)

        # 邻接矩阵序列
        adjacency_matrices = torch.tensor(input_seq['adjacency_matrices'], dtype=torch.float32)

        # 边特征张量
        edge_features = []
        for i in range(self.seq_length):
            edge_feature_matrix = input_seq['edge_features'][i]  # (num_agents, num_agents)
            # 将每个边特征字典转换为向量
            edge_feature_tensor = []
            for edge_list in edge_feature_matrix:
                edge_feature_row = []
                for edge in edge_list:
                    edge_vector = [edge['distance'], edge['relative_angle'], edge['snr'], edge['p_los']]
                    edge_feature_row.append(edge_vector)
                edge_feature_tensor.append(edge_feature_row)
            edge_feature_tensor = torch.tensor(edge_feature_tensor, dtype=torch.float32)
            edge_features.append(edge_feature_tensor)
        edge_features = torch.stack(edge_features)

        # 目标值：未来的动作（转向角和速度）
        target_actions = torch.tensor(target_seq['actions'], dtype=torch.float32)  # (pred_length, num_agents, 2)

        # 通信受限模拟：随机遮盖部分历史轨迹序列，模拟信息丢失
        # 可以在 node_features 中对 history_sequences 进行遮盖
        for i in range(self.seq_length):
            for agent_id in range(len(node_features[i]['history_sequences'])):
                if np.random.rand() < 0.1:  # 假设有 10% 的概率信息丢失
                    node_features[i]['history_sequences'][agent_id] = None  # 或者用特殊标记替代

        # 构建掩码，用于模型中处理缺失信息
        mask = self._create_mask(node_features)

        return node_features, adjacency_matrices, edge_features, target_actions, mask

    def _create_mask(self, node_features):
        """
        根据 node_features 创建掩码，标记缺失的信息。

        返回：
        - mask：形状 (seq_length, num_agents, ...)
        """
        seq_length = len(node_features)
        num_agents = len(node_features[0]['history_sequences'])
        mask = np.ones((seq_length, num_agents), dtype=np.float32)
        for i in range(seq_length):
            for agent_id in range(num_agents):
                if node_features[i]['history_sequences'][agent_id] is None:
                    mask[i, agent_id] = 0.0  # 标记为缺失
        mask = torch.tensor(mask, dtype=torch.float32)
        return mask

