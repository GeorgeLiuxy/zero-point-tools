# PPO 连续控制实验（Ant-v4）

尽管目录名为 `mountain_bike`，当前代码实际使用 Stable-Baselines3 的 PPO 算法训练 Gymnasium MuJoCo `Ant-v4` 环境，目标是学习四足智能体的连续动作控制。

## 功能

- 使用 PPO 训练连续控制策略。
- 通过 `EvalCallback` 定期评估并保存最佳模型。
- 保存 TensorBoard 日志和 `evaluations.npz` 评估记录。
- 加载 `best_model.zip` 进行可视化测试。

## 安装

```bash
pip install "gymnasium[mujoco]" stable-baselines3 tensorboard numpy
```

MuJoCo 的安装方式可能随 Gymnasium 版本变化，请确保本机图形和物理仿真依赖可用。

## 运行

```bash
cd AI/mountain_bike
python main.py
```

输出通常位于脚本配置的日志目录和最佳模型目录。训练步数、评估频率、是否渲染及模型路径均在 `main.py` 中配置。

## 注意事项

- 连续控制训练耗时较长，建议使用独立环境并记录 Stable-Baselines3、Gymnasium 和 MuJoCo 版本。
- 在无桌面环境运行时应关闭渲染。
- `EvalCallback` 默认写出 `logs/evaluations.npz`，但当前脚本末尾读取的是 `logs/evaluations.csv`，绘图前需要改为读取 NPZ 或先完成格式转换。
- 若需要 `env.render()` 显示画面，创建环境时还应按当前 Gymnasium API 设置合适的 `render_mode`。
