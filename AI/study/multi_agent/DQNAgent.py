import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, gamma=0.95, learning_rate=0.001):
        self.state_size = state_size  # 状态空间大小
        self.action_size = action_size  # 动作空间大小
        self.epsilon = epsilon  # 探索率
        self.epsilon_min = epsilon_min  # 最小探索率
        self.epsilon_decay = epsilon_decay  # 探索率衰减
        self.gamma = gamma  # 折扣因子
        self.learning_rate = learning_rate  # 学习率

        self.memory = deque(maxlen=2000)  # 经验回放池
        self.model = self.build_model()  # 创建Q值网络
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def build_model(self):
        """构建DQN神经网络"""
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),  # 输入层
            nn.ReLU(),
            nn.Linear(24, 24),  # 隐藏层
            nn.ReLU(),
            nn.Linear(24, self.action_size)  # 输出层
        )
        return model

    def act(self, state):
        """根据当前状态选择动作"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # 随机选择动作（探索）
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)  # 计算Q值
        return torch.argmax(q_values).item()  # 返回Q值最大的动作（利用）

    def remember(self, state, action, reward, next_state, done):
        """将经历存入经验回放池"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """从经验回放池中采样，更新Q值"""
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)  # 动作是整数类型
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        # 获取当前Q值
        q_values = self.model(states)  # q_values的形状应该是 [batch_size, action_size]

        # actions的形状是 [batch_size]，将其转换为 [batch_size, 1]
        actions = actions.unsqueeze(1)  # 变为 [batch_size, 1]

        # 使用 gather 来选择对应动作的 Q 值
        # gather 期望的索引维度和 q_values 一致
        # 打印 q_values 和 actions 的形状
        print(f"q_values shape: {q_values.shape}")  # 应该是 [3, 3]
        print(f"actions shape: {actions.shape}")  # 应该是 [3]
        q_values = torch.gather(q_values, 1, actions)  # 形状应为 [batch_size, 1]

        # squeeze 去掉多余的维度，形状变为 [batch_size]
        q_values = q_values.squeeze(1)

        # 获取下一个状态的最大Q值
        next_q_values = self.model(next_states).max(1)[0].detach()  # 这里的max返回的是每个状态下最大Q值

        # 计算目标Q值
        target = rewards + (self.gamma * next_q_values * (1 - dones))  # 目标 Q 值应该是 [batch_size]

        # 计算损失并反向传播
        loss = nn.MSELoss()(q_values, target)  # 计算均方误差损失
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



# 更新epsilon（探索率衰减）
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


