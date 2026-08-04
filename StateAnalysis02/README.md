# 卫星状态分析 V2

该目录在前一版基础上拆分训练与评估流程，使用 CNN + LSTM 对卫星轨道序列进行四分类。

## 文件

- `train_new.py`：读取训练数据、构造滑动窗口、训练并保存 `satellite_model_v4.keras`。
- `evaluation.py`：加载模型和 scaler，在评估数据上计算指标并绘图。
- `train/`、`evaluation/`：数据目录。
- `saved_model/`：模型与预处理产物。

## 运行

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib seaborn joblib
cd StateAnalysis02
python train_new.py
python evaluation.py
```

## 已知配置差异

训练脚本默认保存 `satellite_model_v4.keras`，而评估脚本当前引用 `satellite_model_v2.keras`。运行评估前应把 `model_save_path` 和 `scaler_save_path` 改为实际训练产物，避免加载错误版本。

输入字段、类别映射和序列长度必须与训练时一致；评估数据不能参与 scaler 拟合。
