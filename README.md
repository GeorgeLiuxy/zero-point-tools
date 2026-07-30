# zero-point-tools

一个面向实际问题的 Python、人工智能与自动化工具集合。

本仓库并不只是爬虫项目，还包含机器学习、深度学习、生成式 AI、多智能体强化学习、时序预测、自然语言处理、计算机视觉，以及浏览器和桌面自动化等多个方向的实验与应用代码。

> 仓库中的子项目相互独立，主要用于算法研究、模型验证、业务原型开发和日常效率工具建设。不同子项目可能使用不同的数据集、模型权重和依赖环境。

## 核心内容

### 1. 多智能体预测与智能决策

- 基于 STGNN 的多机器人未来动作预测。
- 使用 Transformer 编码历史轨迹，使用 CNN 编码局部地图。
- 通过 GCN 与注意力机制建模机器人之间的空间关系和通信约束。
- 结合 DQN、PPO、QMIX、A*、B 样条等方法进行路径规划、避障和动作决策。
- 支持距离、角度、信噪比、LoS 概率等通信特征。

相关目录：

- `AI/agent_prediction`
- `AI/study/MARL`
- `AI/study/multi_agent`
- `AI/mountain_bike`

### 2. 卫星轨道状态分析与预测

- 根据轨道半长轴、时间、经度和历史控制状态预测卫星运行状态。
- 使用 CNN、LSTM 和 Attention 建立时序分类模型。
- 支持“未控制、受控保持、升轨、降轨”等控制状态分类。
- 包含单输入、双输入和多任务学习版本，可同时预测轨道数值与控制状态。
- 提供数据生成、清洗、滑动窗口构造、标准化、PCA、训练、推理和评估流程。

相关目录：

- `SatMotion`
- `StateAnalysis01`
- `StateAnalysis02`
- `StateAnalysis_V2`

### 3. 生成式视觉与图像融合

- 基于 GAN 和扩散思想实现红外图像与可见光图像融合。
- 包含时间嵌入、正向加噪、对抗训练、L1 重建损失和梯度惩罚等实验。
- 使用 GAN 从随机噪声生成 MNIST 手写数字。
- 集成 Zero123++，支持从单张物体图片生成一致的多视角图像。
- 支持深度 ControlNet、法线图生成、背景分割与 Gradio 交互界面。

相关目录：

- `AI/Diffusion_GAN`
- `AI/diffusionGan`
- `AI/image_generete`
- `zero123plus-main`

### 4. 自然语言处理与安全检测

- 基于 MobileBERT 的 SQL 注入检测，返回检测结果与置信度。
- 提供 Flask Web 页面，可进行本地交互式检测。
- 基于多语言 BERT 的西班牙语评论情感分析。
- 基于西班牙语 BERT 的多类别文本分类训练流程。
- 包含文本分词、数据集封装、模型训练、早停、评估和权重保存。

相关目录：

- `sql_injection_app`
- `Optimization_prjs/no1`
- `AI/demo/simple-sentiment-analysis.py`

### 5. 计算机视觉与 OCR

- 使用 PaddleOCR 识别中文和英文标签内容、坐标与置信度。
- 根据文字区域生成归一化标签结构模板，并进行模板一致性比对。
- 使用 OpenCV 完成标签区域提取、边缘检测、轮廓分析和图像预处理。
- 使用 ResNet50 对图片进行分类，使用 YOLO 完成目标检测与数量统计。
- 使用 Tesseract OCR 辅助识别自动化任务中的图片验证码。

相关目录：

- `label_recognition`
- `scapers/google_image_anasys.py`
- `measure_pairs`
- `utils/koutu.py`

### 6. 推荐、预测与数据分析

- 使用 LightGBM 构建商户推荐模型，结合用户行为、品牌偏好、地理距离、优惠券和时间特征生成 Top-N 推荐。
- 使用 Prophet、线性回归、多项式回归、MLP 和 LSTM 预测销售额与退款率。
- 使用 SVM、逻辑回归、随机森林和决策树进行心脏病风险分类实验。
- 使用随机森林完成天气数据回归预测。
- 包含 Iris 分类、K-Means 聚类、网格搜索和 MNIST 分类等基础机器学习实验。

相关目录：

- `AI/prj1`
- `student_study`
- `code_repair/JQXX/weather-main`
- `针对心脏病发作的分析及预测(1).ipynb`
- `AI/study`

### 7. 气象人工智能

仓库中收录了 GraphCast 与 GenCast 相关工程代码，用于研究：

- 基于图神经网络的全球中期天气预测。
- 基于扩散模型的集合天气预报。
- 稀疏 Transformer、球面网格和自回归多步预测。
- ERA5 等气象数据的预处理、训练和推理流程。

相关目录：`code_repair/GenCast/graphcast-main`

### 8. 爬虫、自动化与效率工具

爬虫与自动化仍然是本仓库的重要组成部分，但不是全部内容，主要包括：

- 安居客、新闻、职位、短视频和社交平台数据采集。
- 在线学习和视频播放自动化。
- 微信消息与界面自动化。
- 浏览器扩展、远程桌面控制和音频通信实验。
- 视频下载、PDF 合并、PDF 转 Word 和图片处理工具。

相关目录：

- `webscraper`
- `scapers`
- `anjuke_phone_scraper`
- `auto_video_study`
- `auto_youtube`
- `wechat`
- `video_download`
- `remote_desktop_controller`
- `ASK-Comm`
- `utils`

## 主要技术栈

| 方向 | 技术 |
| --- | --- |
| 深度学习 | PyTorch、TensorFlow、Keras、PyTorch Geometric |
| NLP | Hugging Face Transformers、BERT、MobileBERT、NLTK |
| 生成式 AI | Diffusers、Stable Diffusion、ControlNet、GAN、扩散模型 |
| 强化学习 | Stable-Baselines3、Gym/Gymnasium、DQN、PPO、QMIX |
| 计算机视觉 | OpenCV、PaddleOCR、YOLO、ResNet50、SAM |
| 机器学习 | scikit-learn、LightGBM、Prophet、PCA |
| 气象模型 | JAX、Haiku、Jraph、GraphCast、GenCast |
| Web 与自动化 | Flask、Gradio、Selenium、Scrapy、BeautifulSoup、Tkinter |
| 数据处理 | NumPy、Pandas、Matplotlib、Seaborn |

## 目录概览

```text
zero-point-tools/
├── AI/                         # AI、深度学习与强化学习实验
├── SatMotion/                  # 卫星轨道数值与控制状态预测
├── StateAnalysis*/             # 卫星状态分析的不同版本
├── sql_injection_app/          # MobileBERT SQL 注入检测系统
├── label_recognition/          # OCR 标签识别与模板比对
├── Optimization_prjs/          # BERT 文本分类等模型实验
├── student_study/              # 销售额、退款率和时序预测
├── zero123plus-main/           # 单图多视角生成与 ControlNet
├── code_repair/                # 天气模型与算法项目
├── webscraper/                 # Web 数据采集工具
├── scapers/                    # 新闻、图片和职位等爬虫
├── auto_video_study/           # 在线学习自动化
├── wechat/                     # 微信自动化工具
├── video_download/             # 视频下载工具
└── utils/                      # PDF、图片等通用工具
```

## 使用方式

本仓库不是一个统一安装、统一启动的单体应用。建议进入具体子项目后，根据其依赖文件和代码入口单独创建环境。

```bash
git clone git@github.com:GeorgeLiuxy/zero-point-tools.git
cd zero-point-tools

# 示例：进入带有独立依赖文件的 MNIST GAN 子项目
cd AI/image_generete

# 安装该子项目依赖并运行
pip install -r requirements.txt
python main.py
```

对于没有独立 `requirements.txt` 的实验，可根据源码中的导入项安装对应依赖。训练类项目通常还需要准备数据集、预训练模型或 GPU 环境。

## 模型与数据说明

- 仓库包含部分 `.pth`、`.h5`、`.keras`、`.safetensors`、`.pkl` 等模型或预处理产物。
- 部分模型会在首次运行时从 Hugging Face、PaddleOCR 或其他模型平台下载权重。
- 多智能体、卫星预测和业务预测项目包含示例数据或仿真数据。
- 大型模型和数据文件建议后续迁移至 Git LFS 或独立的模型存储服务。

## 项目状态

本仓库同时包含可运行应用、研究原型、算法复现和学习实验：

- 部分项目具备完整的数据处理、训练、保存、推理和评估流程。
- 部分目录用于保存不同阶段的实验版本，接口和数据格式可能不同。
- 个别目录仍是原型或占位实现，使用前请先检查入口文件、模型路径和依赖配置。

## 兼容性提醒

当前 Git 历史中存在少量名称末尾带空格的路径，例如 `StateAnalysis ` 和 `remote_desktop_controller/requirements.txt `。这类路径在 Windows 上无法正常检出，建议后续在 Linux 或 WSL 环境中完成重命名并提交。

## 第三方模型与许可

本仓库使用或集成了多个开源框架、预训练模型和研究项目。使用以下内容前，请分别确认其许可证、模型权重许可和数据集使用条件：

- Zero123++ 与相关 ControlNet 模型。
- GraphCast、GenCast 及相关气象数据。
- Hugging Face 上的 BERT、MobileBERT 等预训练模型。
- YOLO、ResNet、PaddleOCR、SAM 等视觉模型。

具体许可说明以各子项目中的 `README`、`LICENSE` 和模型发布页面为准。
