# AI 临时测试脚本

该目录用于保存小规模算法验证，不是正式应用模块。

- `test.py`：MNIST CNN 训练及损失/过拟合现象观察。
- `2_cplex(1).py`：数学优化/CPLEX 相关测试。

运行前请分别检查脚本导入项和本地数据配置：

```bash
cd AI/test
python test.py
```

MNIST 脚本通常需要 TensorFlow/Keras、NumPy 和 Matplotlib；CPLEX 脚本需要 IBM CPLEX Python 环境及相应授权。两个实验没有共享依赖或统一入口。
