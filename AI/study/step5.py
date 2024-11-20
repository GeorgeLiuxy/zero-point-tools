import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# 创建环境，并设置 render_mode
env = gym.make('CartPole-v1', render_mode='human')  # 可以改为 'rgb_array' 或 'human'

# 将环境包装为向量化环境
env = DummyVecEnv([lambda: env])

# 初始化DQN模型
model = DQN('MlpPolicy', env, verbose=1, learning_rate=0.001, buffer_size=50000, exploration_fraction=0.1, exploration_final_eps=0.01)

# 训练模型
model.learn(total_timesteps=10000)

# 保存训练好的模型
model.save("dqn_cartpole")

# 评估训练结果（可视化）
state = env.reset()
for _ in range(1000):
    action, _states = model.predict(state)  # 模型选择动作
    state, reward, done, info = env.step(action)  # 执行动作
    env.render()  # 渲染环境（可视化）

    if done:
        break

# 关闭环境
env.close()
