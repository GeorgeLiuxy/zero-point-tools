# 西班牙语 BERT 多类别文本分类

该项目使用预训练模型 `dccuchile/bert-base-spanish-wwm-uncased` 对西班牙语文本进行多类别分类。代码包含数据加载、标签映射、训练、验证、TensorBoard 记录、早停和模型保存。

## 文件

- `load_data.py`：读取并预处理 `data/train.csv`。
- `model1.py`：BERT 分类模型、训练和验证逻辑。
- `match.py`：训练/匹配流程入口。
- `data/`：训练数据及标签映射文件。

## 安装

```bash
pip install torch transformers pandas numpy scikit-learn tensorboard tqdm
```

首次运行会从 Hugging Face 下载预训练权重，需要网络连接和足够磁盘空间。

## 运行

```bash
cd Optimization_prjs/no1
python match.py
```

## 配置与输出

- 当前分类头按 45 个标签设计，修改标签集时应同步调整 `num_labels` 和标签映射。
- `match.py` 中设备配置偏向 CUDA；没有 GPU 时需改为 CPU。
- 输出包括 TensorBoard 日志、训练指标和模型检查点，实际路径以脚本配置为准。

## 注意事项

训练前请核对 CSV 字段、缺失值处理、标签编码和训练/验证划分。模型效果取决于数据分布，不能仅以训练集表现判断真实泛化能力。
