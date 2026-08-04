# MNIST GAN 数字生成

一个使用 PyTorch 编写的全连接生成对抗网络，用随机噪声生成 MNIST 风格的手写数字。项目适合演示生成器、判别器交替训练和生成结果可视化。

## 运行

```bash
cd AI/image_generete
pip install -r requirements.txt
python main.py
```

首次运行时 Torchvision 会下载 MNIST 数据集。脚本将训练 GAN，并通过 Matplotlib 展示或保存生成样例，具体轮数、批大小、隐变量维度和设备配置可在 `main.py` 中调整。

## 依赖

- PyTorch
- Torchvision
- Matplotlib

该实现用于学习和实验，不包含模型服务接口或生成质量评估指标。
