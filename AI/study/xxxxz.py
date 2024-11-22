import pygame
import numpy as np
import random

from AI.study.xxxx import DQNAgent


class MultiAgentGridEnvironment:
    def __init__(self, width, height, num_agents, obstacles, goals):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.grid = np.zeros((height, width), dtype=int)
        self.obstacles = obstacles
        self.goals = goals
        self.agents = [(random.randint(0, width-1), random.randint(0, height-1)) for _ in range(num_agents)]

        # 设置障碍物
        for (x, y) in obstacles:
            self.grid[y, x] = 1  # 障碍物

    def reset(self):
        """重置环境"""
        self.agents = [(random.randint(0, self.width-1), random.randint(0, self.height-1)) for _ in range(self.num_agents)]
        return self.get_state()

    def get_state(self):
        """获取环境的状态"""
        state = []
        for agent in self.agents:
            state.append(agent)
        return state

    def step(self, actions):
        """根据智能体的动作更新环境"""
        rewards = []
        done = [False] * self.num_agents

        for i, action in enumerate(actions):
            x, y = self.agents[i]
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
                x, y = self.agents[i]
                rewards.append(-1)  # 撞墙，负奖励
                done[i] = False
            else:
                self.agents[i] = (x, y)
                if (x, y) == self.goals[i]:
                    rewards.append(1)  # 到达目标，正奖励
                    done[i] = True
                else:
                    rewards.append(0)  # 未完成任务，没有奖励
        return self.get_state(), rewards, done

    def render(self, screen):
        """通过 Pygame 渲染环境"""
        cell_size = 50  # 每个格子的大小
        screen.fill((255, 255, 255))  # 清空屏幕，填充白色

        # 绘制障碍物
        for (x, y) in self.obstacles:
            pygame.draw.rect(screen, (0, 0, 0), (x * cell_size, y * cell_size, cell_size, cell_size))

        # 绘制目标位置
        for i, (goal_x, goal_y) in enumerate(self.goals):
            pygame.draw.circle(screen, (0, 255, 0), (goal_x * cell_size + cell_size // 2, goal_y * cell_size + cell_size // 2), cell_size // 4)

        # 绘制智能体
        for i, (x, y) in enumerate(self.agents):
            pygame.draw.circle(screen, (255, 0, 0), (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2), cell_size // 4)

        pygame.display.update()  # 更新屏幕


# 在这里加入 DQNAgent 类和其他代码...

# 初始化 Pygame
pygame.init()
screen_width = 7 * 50  # 7格宽
screen_height = 7 * 50  # 7格高
screen = pygame.display.set_mode((screen_width, screen_height))

# 初始化环境和智能体
env = MultiAgentGridEnvironment(width=7, height=7, num_agents=2, obstacles=[(3, 3), (3, 4), (3, 5)], goals=[(6, 6), (5, 5)])
agents = [DQNAgent(state_size=2, action_size=4) for _ in range(env.num_agents)]

# 训练循环
episodes = 1000
batch_size = 32
for e in range(episodes):
    state = env.reset()
    done = [False] * env.num_agents
    total_rewards = [0] * env.num_agents

    while not all(done):
        # 输出当前状态
        print(f"Step {e+1}: Current State: {state}")

        # 获取每个智能体的动作
        actions = [agent.act(state[i]) for i, agent in enumerate(agents)]
        print(f"Step {e+1}: Actions taken by agents: {actions}")

        # 执行动作并获得下一个状态、奖励和终止标志
        next_state, rewards, done = env.step(actions)
        print(f"Step {e+1}: Next State: {next_state}, Rewards: {rewards}, Done: {done}")

        # 记住经验
        for i, agent in enumerate(agents):
            agent.remember(state[i], actions[i], rewards[i], next_state[i], done[i])
            print(f"Agent {i} remembers: state={state[i]}, action={actions[i]}, reward={rewards[i]}, next_state={next_state[i]}, done={done[i]}")

        # 让每个 agent 学习并更新 Q 网络
        for agent in agents:
            agent.replay(batch_size)

        # 更新状态和总奖励
        state = next_state
        total_rewards = [total_rewards[i] + rewards[i] for i in range(env.num_agents)]
        print(f"Step {e+1}: Total Rewards: {total_rewards}")

        # 渲染环境
        env.render(screen)

    print(f"Episode {e+1}/{episodes}, Total Rewards: {total_rewards}")

pygame.quit()
