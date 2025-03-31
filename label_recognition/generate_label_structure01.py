from paddleocr import PaddleOCR
from PIL import Image
import json

# 初始化 OCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False)

# 加载图像
template_path = "cropped_motor_label.png"
image = Image.open(template_path)
w, h = image.size  # 获取图像宽高

# 执行 OCR
ocr_result = ocr.ocr(template_path, cls=True)

# 提取归一化结构框
normalized_structure = []
for line in ocr_result[0]:
    coords, (text, score) = line
    x_coords = [pt[0] for pt in coords]
    y_coords = [pt[1] for pt in coords]
    x1, y1 = min(x_coords), min(y_coords)
    x2, y2 = max(x_coords), max(y_coords)
    norm_box = [
        round(x1 / w, 4),
        round(y1 / h, 4),
        round(x2 / w, 4),
        round(y2 / h, 4)
    ]
    normalized_structure.append({
        "text": text,
        "score": round(score, 3),
        "rel_box": norm_box
    })

# 保存结构模板为 JSON
with open("food_label_structure_normalized_01.json", "w", encoding="utf-8") as f:
    json.dump(normalized_structure, f, ensure_ascii=False, indent=2)

print("✅ 模板结构提取完毕，已保存为 JSON 文件：food_label_structure_normalized_01.json")
