# 卫星状态分析扩展版

该目录汇集卫星状态分类的多版实现，涵盖数据生成、CNN-LSTM、注意力模型、独立预测和评估流程。

## 入口

| 文件 | 说明 |
| --- | --- |
| `data_generate.py`、`generate_data.py` | 生成或可视化卫星仿真数据 |
| `train.py` | CNN + LSTM 时间序列训练，默认保存 v3 模型 |
| `train01.py` | CNN + LSTM + MultiHeadAttention 版本，默认保存 v4 模型 |
| `predict.py` | 加载模型/scaler 并把预测写回 CSV |
| `evaluation.py` | 加载模型进行评估和可视化 |
| `test.py` | Plotly 等轨迹可视化实验 |
| `st1.py` | 独立的销售/退款回归实验，与卫星任务无直接关系 |

## 环境与运行

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib seaborn joblib plotly scipy
cd StateAnalysis_V2
python train.py
python predict.py
python evaluation.py
```

运行前必须核对每个脚本顶部的模型版本、scaler 路径、数据目录和序列长度。当前 `train.py`、`train01.py`、`predict.py`、`evaluation.py` 分别引用 v2/v3/v4 等不同文件名，不保证默认配置互相衔接。

## 建模注意事项

`train01.py` 的扩展输入涉及历史 `Category`。如果在预测时无法获得同等来源的真实历史标签，或切分方式让未来标签进入输入，可能产生标签泄漏。部署前应明确可用特征的时间边界，并使用严格的时间切分验证。
