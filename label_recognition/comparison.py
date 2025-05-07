import cv2
from paddleocr import PaddleOCR, draw_ocr

# OCR初始化（CPU模式，单线程）
ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False, cpu_threads=1)

# 模板定义（电机标签，根据实际情况调整）
motor_template = {
    '产品名称': ((20, 100), (300, 130)),
    '型号规格': ((20, 130), (300, 160)),
    '额定功率': ((20, 160), (300, 190)),
    '额定电压': ((20, 190), (300, 220)),
    '制造日期': ((20, 310), (300, 340)),
}

# 标签匹配函数
def match_label_with_template(image_path, template, result_path='match_result.png'):
    result = ocr.ocr(image_path, cls=True)
    image = cv2.imread(image_path)

    match_results = {}
    # 对模板中的每个字段进行校验
    for field, (top_left, bottom_right) in template.items():
        x1, y1 = top_left
        x2, y2 = bottom_right
        matched = False
        matched_text = ""

        for line in result[0]:
            coords, (text, confidence) = line
            cx = [point[0] for point in coords]
            cy = [point[1] for point in coords]
            text_x, text_y = sum(cx) / 4, sum(cy) / 4  # 文本中心点位置

            # 判断OCR识别的文本中心是否在模板规定区域内
            if x1 <= text_x <= x2 and y1 <= text_y <= y2:
                matched = True
                matched_text = text
                break

        match_results[field] = (matched, matched_text)

        # 在图片上绘制模板区域和判定结果
        color = (0, 255, 0) if matched else (0, 0, 255)
        cv2.rectangle(image, top_left, bottom_right, color, 2)
        cv2.putText(image, field, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 打印结果
    for field, (matched, matched_text) in match_results.items():
        status = ' 匹配' if matched else '❌ 不匹配'
        print(f"{field}: {status} - 识别内容: {matched_text}")

    cv2.imwrite(result_path, image)
    print(f"匹配结果图片已保存至：{result_path}")

# 示例调用
if __name__ == '__main__':
    match_label_with_template('testImage/motor_label_with_barcode.png', motor_template, 'motor_label_match_result.png')
