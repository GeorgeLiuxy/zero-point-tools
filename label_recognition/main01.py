from paddleocr import PaddleOCR
import cv2
import pandas as pd

# 初始化 OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

# 设置路径
image_path = 'cropped_motor_label.png'

# 读取图像并识别
result = ocr.ocr(image_path, cls=True)

# 整理结果
info_blocks = []
for line in result[0]:
    coords, (text, score) = line
    x_coords = [pt[0] for pt in coords]
    y_coords = [pt[1] for pt in coords]
    x1, y1 = min(x_coords), min(y_coords)
    x2, y2 = max(x_coords), max(y_coords)
    info_blocks.append({
        "识别内容": text,
        "置信度": round(score, 3),
        "区域坐标": f"({int(x1)}, {int(y1)}) ~ ({int(x2)}, {int(y2)})"
    })

# 打印或保存表格
df = pd.DataFrame(info_blocks)
print(df)
df.to_csv("识别结果.csv", index=False)
