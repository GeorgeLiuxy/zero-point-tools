# 基于神经网络的卫星运动行为分析

## 项目概述

本项目旨在利用基于神经网络（如 CNN、LSTM 和 Attention）的智能算法对卫星的运动行为进行分析与预测。通过轨道相关数据输入，可以训练模型对卫星的控制状态进行预测，并最终实现对新数据的行为分析和预测结果导出。

本项目主要实现以下功能：
1. **数据分析**：基于现有神经网络模型，分析卫星的运动行为。
2. **模型训练与评估**：通过数据集训练模型，并对模型的性能进行评估，确保准确率达到预期要求。
3. **新数据预测**：扩展现有代码，实现对训练完成模型的调用，支持新数据输入的预测。
4. **结果输出**：将预测结果导出为文件，便于进一步分析。

---

## 当前状态

- **已实现功能**：
    - 数据预处理与处理流程。
    - 基于 CNN、LSTM、Attention 结合的模型架构进行训练。
    - 在训练数据集上的模型性能评估。

- **现有代码的局限性**：
    - 无法保存训练完成的模型文件，供后续调用。
    - 缺少对新数据的预测功能。
    - 无法将预测结果输出为文件。

---

## 项目目标

### 1. 实现基于训练模型的卫星行为预测
- 增加保存模型的功能。
- 实现加载已保存的模型，对新数据进行预测。
- 支持用户根据需求选择不同模型（CNN+LSTM+Attention、LSTM、RNN 等）进行预测。

### 2. 导出预测结果
- 将预测结果保存为结构化文件（如 CSV 或 Excel 文件）。

### 3. 保证模型准确率
- 确保模型在现有内部评价体系下达到 **95%** 的准确率。

---

## 运行脚本
## 运行示例
## 训练单输入 LSTM 模型
```bash
python main.py train --model_type lstm --data_path 10294.csv --seq_length 10 --model_name single_input_lstm.h5
```
## 训练双输入 RNN 模型[10294.csv](10294.csv)
```bash
python main.py train --model_type rnn --data_path 10294.csv --seq_length 10 --model_name dual_input_rnn.h5 --dual_input
```
## 预测新数据（单输入模型）[59020.csv](59020.csv)
```bash
python main.py predict --model_type lstm --model_name single_input_lstm.h5 --new_data_path 59020.csv --output_path single_input_predictions.csv
```
## 预测新数据（双输入模型）
```bash
python main.py predict --model_type rnn --model_name dual_input_rnn.h5 --new_data_path 59020.csv --output_path dual_input_predictions.csv --dual_input
```