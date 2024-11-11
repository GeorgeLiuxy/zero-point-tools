import random

import numpy as np
import pickle


class RLAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.actions = actions
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def get_state_key(self, state):
        return tuple(state)

    def get_action(self, state):
        state_key = self.get_state_key(state)
        if random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions), np.random.uniform(0, 2*np.pi), np.random.choice([0, 1])

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.actions))

        # 假设返回的动作包含 速度，角度，路径选择
        speed_angle = np.argmax(self.q_table[state_key])  # 或者自定义
        angle = np.random.uniform(0, 2*np.pi)  # 模拟的角度选择
        path_choice = np.random.choice([0, 1])  # 0: 使用A*路径，1: 动态调整路径

        return speed_angle, angle, path_choice


    def update(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.actions))
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(len(self.actions))

        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward + self.discount_factor * self.q_table[next_state_key][best_next_action]
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.learning_rate * td_error

    def save_model(self, filename='q_table.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filename}")

    def load_model(self, filename='q_table.pkl'):
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Q-table loaded from {filename}")
        except FileNotFoundError:
            print(f"No saved model found. Starting fresh.")
            self.q_table = {}

