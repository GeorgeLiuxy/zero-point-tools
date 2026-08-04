# AI 与智能算法实验

该目录集中保存机器学习、深度学习、强化学习、生成式模型和多智能体预测实验。各子目录相互独立，没有统一的启动入口；运行前应进入对应目录并查看其 README、数据路径和依赖文件。

## 子项目导航

| 目录 | 内容 | 当前状态 |
| --- | --- | --- |
| `agent_prediction` | STGNN 多智能体动作预测，以及 DQN、A*、B 样条避障实验 | 研究原型，包含示例数据 |
| `Diffusion_GAN` | 红外与可见光融合的早期 GAN/扩散结构 | 未接通数据集，不可直接完整训练 |
| `diffusionGan` | 红外与可见光图像融合的多版 PyTorch 实现 | 可实验，需准备成对图片 |
| `image_generete` | 基于全连接 GAN 的 MNIST 数字生成 | 可直接运行，首次下载 MNIST |
| `mountain_bike` | Gymnasium `Ant-v4` 环境上的 PPO 训练与评估 | 可运行，目录名与实际任务不一致 |
| `prj1` | 基于 LightGBM 的用户—商户 Top-N 推荐 | 可实验，包含样例 CSV |
| `demo` | 情感分类、时序卷积、多智能体与可视化小实验 | 代码草稿集合 |
| `study` | 经典机器学习、DQN、MARL、QMIX 学习代码 | 学习与复现实验 |
| `test` | MNIST CNN 和优化问题测试脚本 | 临时实验 |

## 使用建议

1. 为不同子项目分别创建虚拟环境，避免 PyTorch、TensorFlow、Gymnasium 等版本互相影响。
2. 优先使用子目录已有的 `requirements.txt`；没有依赖文件时，根据 README 中列出的包安装。
3. 训练脚本普遍会读取本地数据或下载预训练权重，运行前先检查代码顶部及 `main` 区域中的路径、设备和超参数。
4. `.pth`、`.h5`、`.keras`、`.pkl` 等文件属于模型或预处理产物，不应默认视为跨版本兼容。

## 与 AI 相关的能力

- 图神经网络、Transformer、CNN 与注意力机制的多智能体时空建模。
- DQN、PPO、QMIX 等强化学习方法。
- GAN、扩散过程和图像融合。
- BERT 文本分类与情感分析。
- LightGBM 推荐排序及经典机器学习算法。
