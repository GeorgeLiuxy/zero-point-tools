import json
from paddleocr import PaddleOCR
from PIL import Image

# ==== 参数配置 ====
TEMPLATE_JSON_PATH = "food_label_structure_normalized.json"
TARGET_IMAGE_PATH = "testImage/food_label_with_barcode.png"
MATCH_TOLERANCE = 0.03      # 单个框坐标容忍偏差
MATCH_THRESHOLD = 0.8       # 结构相似度阈值（80%）

# ==== 初始化 OCR ====
ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False)


# ==== 提取归一化结构框 ====
def extract_normalized_boxes(image_path):
    image = Image.open(image_path)
    w, h = image.size
    ocr_result = ocr.ocr(image_path, cls=True)

    norm_boxes = []
    for line in ocr_result[0]:
        coords, (text, score) = line
        x_coords = [pt[0] for pt in coords]
        y_coords = [pt[1] for pt in coords]
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)
        norm_boxes.append([
            round(x1 / w, 4),
            round(y1 / h, 4),
            round(x2 / w, 4),
            round(y2 / h, 4)
        ])
    return norm_boxes


# ==== 判断两个框是否相似 ====
def is_box_similar(box1, box2, tolerance=MATCH_TOLERANCE):
    return all(abs(a - b) <= tolerance for a, b in zip(box1, box2))


# ==== 主结构比对逻辑 ====
def compare_structure(template_path, target_image_path):
    # 读取模板结构
    with open(template_path, "r", encoding="utf-8") as f:
        template_boxes = json.load(f)

    template_coords = [b["rel_box"] for b in template_boxes]
    target_coords = extract_normalized_boxes(target_image_path)

    min_len = min(len(template_coords), len(target_coords))
    if min_len == 0:
        return 0.0, 0, len(template_coords), len(target_coords), False

    match_count = 0
    for i in range(min_len):
        if is_box_similar(template_coords[i], target_coords[i]):
            match_count += 1

    similarity = match_count / max(len(template_coords), len(target_coords))
    is_match = similarity >= MATCH_THRESHOLD

    return similarity, match_count, len(template_coords), len(target_coords), is_match


# ==== 执行对比 ====
similarity, match_count, temp_total, targ_total, result = compare_structure(
    TEMPLATE_JSON_PATH,
    TARGET_IMAGE_PATH
)

# ==== 输出结果 ====
print("📌 模板框数:", temp_total)
print("📌 当前图识别框数:", targ_total)
print("📌 匹配成功框数:", match_count)
print("📌 结构相似度:", round(similarity, 2))
print(" 是同一模板" if result else "❌ 不是同一模板")
