# 卫星状态分析 V1

该项目使用卫星轨道时间序列训练 CNN + LSTM 多分类模型，预测未控制、受控保持、升轨和降轨等控制状态，并保存标准化器、Keras 模型和预测结果。

## 主要文件

- `generate_data.py`、`generate_data_new.py`：生成训练、评估和预测用仿真 CSV。
- `train_01.py`：较完整的训练、评估、模型保存和新数据预测流程。
- `train_new.py`：另一版 CNN + LSTM 训练实现。
- `train/`、`evaluation/`、`prediction/`：分阶段数据目录。
- `saved_model/`：Keras 模型及 scaler 等产物。

## 安装与运行

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib seaborn joblib
cd StateAnalysis01
python train_01.py
```

如需重新生成数据，可先运行相应的 `generate_data*.py`。脚本默认从当前目录的相对路径读取数据，因此应在本目录启动。

## 数据与输出

输入 CSV 通常包含 `Timestamp`、轨道数值特征和 `Category` 标签。输出包括 `.keras` 模型、`scaler.pkl`、评估图和带预测列的 CSV，具体文件名以脚本中的路径变量为准。

## 注意事项

不同训练脚本对字段和数据目录的约定略有差异。使用真实数据前应固定类别映射、按时间划分训练/验证集，并确保 scaler 只在训练集上拟合。
