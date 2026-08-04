# 机器学习与强化学习学习代码

该目录收录从经典机器学习到深度强化学习、多智能体强化学习的学习与复现实验。

## 内容导航

- Iris 分类：逻辑回归、SVM、决策树等基础分类方法。
- K-Means 聚类与 GridSearch 超参数搜索。
- MNIST 全连接或卷积神经网络分类。
- CartPole DQN 强化学习实验。
- `MARL/`：多智能体强化学习示例，包含独立说明。
- `MARL/QMIX/`：QMIX 实现与说明。
- `multi_agent/`：多智能体 DQN/协作环境实验。

## 使用

每个脚本均为独立入口，先确认所需数据和环境名称后运行：

```bash
cd AI/study
python <脚本名>.py
```

常见依赖包括 NumPy、Pandas、Matplotlib、scikit-learn、PyTorch、TensorFlow、Gym/Gymnasium。强化学习环境的 API 在不同版本间变化较大，若出现 `reset`/`step` 返回值不兼容，应使用与脚本编写时期匹配的版本或调整接口。

该目录面向学习与算法对比，不提供统一包接口。
