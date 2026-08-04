# 多智能体动作预测与避障实验

本项目研究多机器人场景中的未来动作预测和路径避障。核心思路是用时空图神经网络（STGNN）联合编码机器人历史状态、局部地图以及机器人之间的空间关系，预测速度和转向；扩展版本又加入 DQN、A* 和 B 样条路径平滑。

## 主要实现

- Transformer 编码历史轨迹和状态序列。
- CNN 编码机器人周围的局部栅格地图。
- GCN 与注意力层建模机器人间距离、角度、信噪比和通信质量等关系。
- `new_code` 中包含基于 DQN 的决策、A* 路径搜索和 B 样条平滑实验。

## 目录说明

| 路径 | 说明 |
| --- | --- |
| `code/main.py` | STGNN 实验入口 |
| `code/train.py` | 训练与数据处理逻辑 |
| `new_code/main.py` | DQN、A* 与路径平滑组合实验 |
| `new_code/STGNN.py` | 时空图网络结构 |
| `new_code/DQN.py` | DQN 网络与智能体封装 |
| `new_code/MADDPGAgent.py` | 多智能体控制封装；当前实现以多个 DQN 智能体为主 |
| `old/` | 早期版本，供对照使用 |
| `simulation_data*.pkl` | 仿真样例数据 |

## 环境

基础依赖为 Python、NumPy、PyTorch 和 Matplotlib；部分实现还会使用图网络或科学计算相关包。可先进入具体版本目录安装其依赖：

```bash
cd AI/agent_prediction/new_code
pip install -r requirements.txt
python main.py
```

旧版入口可在对应目录用相同方式运行。建议从脚本所在目录启动，以保证相对路径能够找到数据文件。

## 输入与输出

- 输入：机器人历史状态、局部地图、邻接/通信特征以及仿真环境状态。
- 输出：预测速度、转向动作、规划路径、训练损失及轨迹可视化。

## 注意事项

- 这是研究型代码，不同版本的数据字段、张量维度和模型接口可能不同。
- `MADDPGAgent.py` 的文件名保留了实验阶段命名，当前代码并非标准 MADDPG 实现。
- 训练规模和设备选择需在脚本中调整；使用自定义数据前应先核对 pickle 中的结构。
