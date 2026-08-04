# 标签 OCR、结构生成与模板比对

该目录使用 PaddleOCR、OpenCV 和 Pillow 对产品标签进行文字识别、区域提取、版面结构归一化和模板一致性比对。当前实现由多个独立脚本组成，没有 PyQt/Tkinter 图形界面，也没有使用原设计文档中提到的 Tesseract/SIFT 主流程。

## 主要脚本

| 文件 | 功能 |
| --- | --- |
| `main.py` | 中文 PaddleOCR，输出文字、坐标、置信度和标注图 |
| `main01.py` | 英文 PaddleOCR，并导出 `识别结果.csv` |
| `extact_label_region.py` | 通过 OpenCV 轮廓提取标签区域 |
| `generate_label_structure.py` | 从模板图生成归一化文字框 JSON |
| `generate_label_structure01.py` | 另一模板结构生成版本 |
| `comparison_label_model.py` | 将目标标签的归一化框与模板 JSON 比较 |
| `comparison.py` | OCR 内容/位置匹配和结果绘制实验 |
| `generate_label_template*.py` | 使用 Pillow 生成带文本和条码的标签样例 |
| `xxxx.py` | 与 WADI 数据相关的独立实验，不属于标签主流程 |

## 安装

```bash
pip install paddleocr paddlepaddle opencv-python pillow pandas numpy
```

PaddlePaddle 的 CPU/GPU 安装包应按操作系统和计算环境选择。首次运行 PaddleOCR 可能下载模型权重。

## 运行示例

```bash
cd label_recognition
python main.py
python generate_label_structure.py
python comparison_label_model.py
```

脚本中的图片、模板 JSON 和输出路径目前为硬编码，运行前需改成实际文件。建议流程为：标签区域裁剪 → OCR → 模板结构生成 → 目标结构比对。

## 比对说明

结构比对主要使用归一化文字框位置，能降低不同分辨率的影响，但仍对拍摄角度、透视、旋转、标签变形和 OCR 漏检敏感。生产使用前应补充透视校正、字段级规则、置信度阈值和人工复核机制。
