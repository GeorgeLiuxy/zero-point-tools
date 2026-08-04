# 算法与气象项目归档

该目录集中保存若干相互独立的算法复现、气象预测和代码修复实验。它不是一个统一应用，进入具体子项目后应优先阅读其原始 README 和许可证。

## 子项目

| 路径 | 内容 |
| --- | --- |
| `GenCast/graphcast-main` | GraphCast/GenCast 相关气象 AI 工程与上游说明 |
| `JQXX/weather-main` | 基于随机森林等方法的天气数据回归实验 |
| `LSTM/weather-master` | LSTM 天气预测及 Web 展示项目 |
| `jueceshu_no_need` | 自定义决策树等算法草稿 |

## 使用原则

- 各项目依赖、数据集和运行方式不同，不应在 `code_repair` 根目录统一安装。
- GraphCast/GenCast 通常依赖 JAX、Haiku、气象网格数据和较高计算资源，请以其目录内上游文档为准。
- 气象数据、预训练权重及第三方代码的许可条件可能彼此不同，使用和再发布前需逐项确认。
- `build`、缓存和历史版本目录只作为归档，不应优先于源目录运行。
