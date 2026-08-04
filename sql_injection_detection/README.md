# SQL 注入检测训练工程占位目录

该目录预留了基于 DistilBERT/Transformers 的 SQL 注入检测训练工程结构，包括 `config.py`、数据处理、模型、训练、评估、预测、工具函数和模板文件。

## 当前状态

除模板外，目录中的 Python、配置和模型文件目前均为空文件，因此该项目不能安装、训练或推理。它不是 `../sql_injection_app` 的可用训练代码。

## 预期结构

| 文件 | 预期职责 |
| --- | --- |
| `config.py` | 模型、数据、训练参数和路径配置 |
| `data_preprocess.py` | SQL 样本清洗、标签编码和数据切分 |
| `dataset.py` | PyTorch Dataset/DataLoader |
| `model.py` | DistilBERT 分类模型 |
| `train.py` | 训练、验证和检查点保存 |
| `evaluate.py` | Accuracy、Precision、Recall、F1 等评估 |
| `predict.py` | 单条或批量推理 |
| `utils.py` | 日志、随机种子等工具 |

## 完成建议

要使其可运行，需要先定义合法/注入 SQL 数据格式和标签语义，再实现上述模块、添加 `requirements.txt`、训练/验证/测试隔离、模型版本记录和安全测试集。完成前请使用已有的 `../sql_injection_app` 做本地推理演示。
