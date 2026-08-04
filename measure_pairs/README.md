# 轮廓关键点配对与尺寸测量

该工具使用 OpenCV 和 SciPy 从图片中提取物体轮廓，对轮廓进行平滑，识别峰谷关键点，并按距离和角度约束进行点对匹配，最终导出长度及夹角等测量结果。

## 主要功能

- 灰度化、旋转、模糊、Canny/自适应阈值和形态学处理。
- 轮廓查找及 CubicSpline、Savitzky-Golay 平滑。
- 峰谷检测和候选点配对。
- 计算点对长度、相邻距离以及左右夹角。
- 输出 CSV 和测量结果图。

## 文件

- `measure_pairs.py`：主要实验入口，当前默认读取 `images/11.jpg`。
- `mesure_pairs_*.py`：不同参数和算法阶段的变体。
- `test.py`：图像增强、修复和区域生长等辅助实验。
- `images/`：样例输入与处理结果。

## 安装与运行

```bash
pip install numpy opencv-python scipy pandas matplotlib
cd measure_pairs
python measure_pairs.py
```

更换图片时需修改脚本中的输入路径，并按拍摄比例调整阈值、核大小、峰值间距和点对距离范围。

## 测量说明

脚本输出默认是图像像素尺度。若要得到毫米等物理单位，必须加入已知尺寸标定物并计算像素—实际长度换算；透视畸变明显时还需先做相机标定和透视校正。
