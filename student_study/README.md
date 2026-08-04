# 销售额与退款率预测实验

该目录使用合成订单和 Excel/CSV 业务数据，比较 Prophet、线性回归、多项式回归、MLP 和 LSTM 等方法在月度销售额、订单金额及退款率预测上的表现。

## 文件

- `data_generete.py`：使用 Faker 生成 2018—2023 年订单样例数据。
- `pred_monthly.py`：Prophet 与回归模型的月度预测和可视化。
- `pred_monthly_LSTM.py`：增加 MLP、LSTM 等神经网络模型。
- `pred_monthly_v1.py`：较简化的回归版本。
- `st1.py`：读取 `shanghai.xlsx`，进行退款/销售相关回归分析。
- `orders_2018_2023*.csv`、`shanghai.xlsx`：样例数据。

## 安装与运行

```bash
pip install pandas numpy matplotlib seaborn scikit-learn prophet tensorflow Faker openpyxl
cd student_study
python pred_monthly.py
```

如需重建合成数据：

```bash
python data_generete.py
```

## 注意事项

- 脚本会直接读取固定文件名，运行前需检查 `file_path`。
- 时间序列评估应按时间顺序切分，避免随机切分造成未来数据泄漏。
- 合成数据只适合验证流程，预测结论不能直接用于真实经营决策。
- Prophet 和 TensorFlow 安装较重，建议使用独立虚拟环境。
