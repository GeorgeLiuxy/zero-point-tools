import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
import pandas as pd

# 创建环境
env = gym.make("Ant-v4")

# 初始化模型
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10)

# 定义评估回调
eval_callback = EvalCallback(env, best_model_save_path="./logs/", log_path="./logs/", eval_freq=10000, deterministic=True)

# 训练模型
model.learn(total_timesteps=200000, callback=eval_callback)

# 加载表现最优的模型并测试
model = PPO.load("./logs/best_model")
obs, info = env.reset()
done = False
cumulative_reward = 0

while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    cumulative_reward += reward
    done = terminated or truncated
    env.render()
env.close()
print(f"Total Reward: {cumulative_reward}")

# 绘制奖励变化曲线
results = pd.read_csv("./logs/evaluations.csv")
plt.plot(results["timesteps"], results["mean_reward"])
plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.title("Training Performance Over Time")
plt.show()
