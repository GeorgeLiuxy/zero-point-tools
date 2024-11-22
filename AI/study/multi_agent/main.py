from AI.study.multi_agent.DQNAgent import DQNAgent
from AI.study.multi_agent.MultiAgentPushEnv import MultiAgentPushEnv


def train_multi_agent(env, num_episodes=1000, batch_size=32):
    agents = [DQNAgent(state_size=2, action_size=4) for _ in range(env.num_agents)]
    for episode in range(num_episodes):
        state = env.reset()  # 重置环境
        done = [False] * env.num_agents  # 记录每个智能体是否完成任务
        total_rewards = [0] * env.num_agents  # 记录每个智能体的总奖励

        while not all(done):  # 当所有智能体任务都完成时退出
            actions = []
            for agent_idx in range(env.num_agents):
                action = agents[agent_idx].act(state[agent_idx])  # 选择动作
                actions.append(action)

            next_state, rewards, done = env.step(actions)  # 执行动作并得到反馈

            # 存储经历并进行Q值更新
            for agent_idx in range(env.num_agents):
                agents[agent_idx].remember(state[agent_idx], actions[agent_idx], rewards[agent_idx], next_state[agent_idx], done[agent_idx])
                agents[agent_idx].replay(batch_size)  # 执行Q学习的回放更新

            state = next_state  # 更新状态
            total_rewards = [x + y for x, y in zip(total_rewards, rewards)]  # 累加奖励

        # 输出每100轮的训练情况
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes}, Rewards: {total_rewards}")

# 训练环境
env = MultiAgentPushEnv(width=5, height=5, num_agents=2, obstacles=[(2, 2)], goal=(4, 4), initial_object_pos=(0, 0))
train_multi_agent(env)
