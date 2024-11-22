import numpy as np
import random

class MultiAgentPushEnv:
    def __init__(self, width, height, num_agents, obstacles, goal, initial_object_pos):
        self.width = width  # 网格宽度
        self.height = height  # 网格高度
        self.num_agents = num_agents  # 智能体数量
        self.agents = []
        self.grid = np.zeros((height, width), dtype=int)  # 创建一个空的网格
        self.obstacles = obstacles  # 障碍物位置
        self.goal = goal  # 目标位置
        self.object_pos = initial_object_pos  # 物体初始位置

        # 设置障碍物
        for (x, y) in obstacles:
            self.grid[y, x] = 1  # 障碍物

        # 设置目标
        self.grid[goal[1], goal[0]] = 2  # 目标位置

        # 设置物体
        self.grid[initial_object_pos[1], initial_object_pos[0]] = 3  # 物体位置

        # 设置智能体
        for _ in range(num_agents):
            while True:
                agent_pos = (random.randint(0, width-1), random.randint(0, height-1))
                if self.grid[agent_pos[1], agent_pos[0]] == 0:  # 确保智能体不在障碍物或物体上
                    self.agents.append(agent_pos)
                    break

    def reset(self):
        """重置环境"""
        self.agents = []
        self.grid = np.zeros((self.height, self.width), dtype=int)
        # 重新设置障碍物、目标、物体和智能体
        for (x, y) in self.obstacles:
            self.grid[y, x] = 1
        self.grid[self.goal[1], self.goal[0]] = 2
        self.grid[self.object_pos[1], self.object_pos[0]] = 3
        for _ in range(self.num_agents):
            while True:
                agent_pos = (random.randint(0, self.width-1), random.randint(0, self.height-1))
                if self.grid[agent_pos[1], agent_pos[0]] == 0:
                    self.agents.append(agent_pos)
                    break
        return self.get_state()

    def get_state(self):
        """返回当前状态"""
        return self.agents, self.object_pos

    def step(self, actions):
        """根据智能体的动作更新环境"""
        rewards = []
        done = [False] * self.num_agents

        for i, action in enumerate(actions):
            x, y = self.agents[i]

            # 根据动作更新位置
            if action == 0:  # 上
                y -= 1
            elif action == 1:  # 右
                x += 1
            elif action == 2:  # 下
                y += 1
            elif action == 3:  # 左
                x -= 1

            # 边界检查
            if x < 0 or x >= self.width or y < 0 or y >= self.height or self.grid[y, x] == 1:
                # 撞墙或者碰到障碍物
                x, y = self.agents[i]  # 保持原位
                rewards.append(-1)  # 撞墙，负奖励
                done[i] = False
            else:
                self.agents[i] = (x, y)
                rewards.append(0)  # 无奖励

            # 推物体逻辑
            if (x, y) == self.object_pos:
                new_object_pos = self.push_object(x, y)
                if new_object_pos != self.object_pos:
                    self.object_pos = new_object_pos

            # 检查是否完成任务
            if self.object_pos == self.goal:
                done[i] = True
                rewards[i] = 10  # 到达目标位置，正奖励

        return self.get_state(), rewards, done

    def push_object(self, x, y):
        """如果智能体与物体接触，则推动物体"""
        ox, oy = self.object_pos
        if abs(x - ox) <= 1 and abs(y - oy) <= 1:  # 物体周围的任何一个格子
            # 推物体到相邻位置
            if x > ox: ox += 1
            elif x < ox: ox -= 1
            if y > oy: oy += 1
            elif y < oy: oy -= 1
        return (ox, oy)
