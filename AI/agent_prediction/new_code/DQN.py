import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# DQN 神经网络
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN 代理
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.batch_size = 32

    def act(self, state):
        """选择动作：ε-greedy策略"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        return torch.argmax(self.model(state)).item()

    def remember(self, state, action, reward, next_state, done):
        """记忆回放"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """从记忆中回放并更新Q网络"""
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = self.model(state).clone()
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * torch.max(self.target_model(next_state))
            self.optimizer.zero_grad()
            loss = torch.nn.MSELoss()(self.model(state)[action], target[action])
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        """定期更新目标模型"""
        self.target_model.load_state_dict(self.model.state_dict())
