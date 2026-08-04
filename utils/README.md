# 通用文件与图像工具

该目录保存若干互相独立的小工具，主要用于 CSV 合并、PDF 合并标注和图片轮廓提取。

## 工具列表

### CSV 合并：`fileUtils.py`

读取指定目录下的所有 CSV，保留第一份表头并合并数据。当前脚本底部包含 macOS 硬编码输入路径，运行前必须修改 `folder_path` 和 `output_file`。

```bash
python fileUtils.py
```

### PDF 合并：`pdfcomb.py`

合并 `pdf/` 目录中的 PDF，并在合并文件页面底部叠加标注。依赖由本目录 `requirements.txt` 提供：

```bash
pip install -r requirements.txt
python pdfcomb.py
```

当前标注内容使用合并后文件名，而不是逐页对应的原始文件名；如需追踪原文件，应在合并时同步记录页码范围。

### 图片轮廓提取：`koutu.py`

读取 `image.jpg`，通过 Canny、闭运算和外轮廓填充生成 `extracted_region_canny.jpg`。

```bash
pip install opencv-python numpy
python koutu.py
```

脚本会打开 OpenCV 窗口，无图形桌面环境需移除 `imshow` 相关代码。

## 其他文件

`yolov5su.pt` 是模型权重资产，当前这三个工具没有直接引用它。仓库根目录的 `pdf2word.py` 是独立的 PDF 转 Word 工具。
