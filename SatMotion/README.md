# 卫星运动多任务预测

该项目使用 CNN、LSTM 和自定义 Attention 同时完成两类任务：回归预测轨道数值 `Value`，以及分类预测控制状态 `Category`。目录包含单输入、双输入实验、样例 CSV、已训练模型和预处理产物。

## 入口

| 文件 | 说明 |
| --- | --- |
| `single_input_model.py` | 单序列多任务模型；训练、评估、保存、加载和新数据预测 |
| `dual_input_model.py` | 双输入/扩展多任务模型版本 |
| `test.py` | 数据和模型验证实验 |
| `utils.py` | CSV 合并等数据辅助函数 |
| `demo/` | 时间特征和多输入结构的阶段性实验 |

## 环境

依赖文件名保留为历史拼写 `requitements.txt`：

```bash
cd SatMotion
pip install -r requitements.txt
pip install numpy scikit-learn joblib
```

其中列出了 Pandas、Matplotlib 和 TensorFlow，代码还会使用 NumPy、scikit-learn 和部分版本的 joblib。

## 运行

```bash
python single_input_model.py
# 或
python dual_input_model.py
```

这些脚本没有命令行参数，运行后会直接训练模型、绘制结果并执行新数据预测。

## 数据和产物

- `single_input_model.py` 默认训练数据为 `merged_data.csv`，新数据路径当前写成 `./cnn_data/59020.csv`。
- `dual_input_model.py` 当前默认读取 `./cnn_data/10294.csv` 和 `./cnn_data/59020.csv`。
- 仓库里的样例 CSV 位于目录根部，而不是 `cnn_data/`；运行前需修改路径或建立对应目录。
- 产物包括 `multi_task_model.h5`、`trained_model.keras`、scaler/label encoder 文件和 `new_data_predictions.csv`。

## 注意事项

模型训练和预测接口在多个版本间不完全一致。重新训练时应固定序列长度、特征顺序、类别编码和 scaler，并让推理阶段加载与模型配套的预处理产物。
