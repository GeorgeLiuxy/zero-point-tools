# AI 算法演示与草稿

该目录是多个独立实验脚本的集合，不是单一应用。内容涵盖自然语言处理、时序建模、多智能体强化学习、轨迹与网络结构可视化，以及图片水印处理。

## 代表性内容

- `simple-sentiment-analysis.py`：基于预训练 BERT/Transformers 的情感分析示例。
- Temporal Convolution Network 相关脚本：时序卷积结构实验。
- 多智能体、强化学习和时空网络脚本：算法结构与仿真验证。
- 可视化脚本：轨迹、网络或训练结果绘制。
- 水印脚本：图片水印处理实验。

## 使用方式

先打开目标脚本，确认其数据路径、模型名称和输出路径，再单独运行：

```bash
cd AI/demo
python <脚本名>.py
```

依赖随脚本而异，可能包括 PyTorch、Transformers、NumPy、Pandas、Matplotlib、NetworkX、OpenCV 和 Pillow。

## 状态说明

这里的文件主要用于快速验证想法，接口、参数和数据格式没有统一。建议将计划长期维护的实验迁移到独立子目录，并补充依赖锁定、样例数据和可复现配置。
