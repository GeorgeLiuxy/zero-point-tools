import torch
import torch.nn as nn
import torch.optim as optim

from new_code.DQN import DQNAgent


# 假设有多个智能体
class MADDPGAgent:
    def __init__(self, state_size, action_size, num_agents):
        self.num_agents = num_agents
        self.agents = [DQNAgent(state_size, action_size) for _ in range(num_agents)]  # 为每个智能体创建一个代理
        self.target_agents = [DQNAgent(state_size, action_size) for _ in range(num_agents)]  # 目标代理
        self.optimizers = [optim.Adam(agent.model.parameters(), lr=0.001) for agent in self.agents]

    def act(self, states):
        """每个智能体根据当前状态选择动作"""
        actions = []
        for i in range(self.num_agents):
            actions.append(self.agents[i].act(states[i]))
        return actions

    def replay(self):
        """每个智能体根据回放更新Q网络"""
        for agent in self.agents:
            agent.replay()

    def update_target_models(self):
        """定期更新目标代理"""
        for i in range(self.num_agents):
            self.agents[i].update_target_model()
