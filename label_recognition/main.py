import cv2
from paddleocr import PaddleOCR, draw_ocr

# 初始化PaddleOCR模型，支持中文识别
ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False, cpu_threads=1)

# 标签识别函数（内容+位置）
def recognize_label_with_positions(image_path, output_path='result.png'):
    # 执行OCR识别
    result = ocr.ocr(image_path, cls=True)

    # 输出识别结果与位置
    for line in result[0]:
        coords, (text, confidence) = line
        print(f"识别文本: {text}\n坐标位置: {coords}\n置信度: {confidence}\n{'-'*30}")

    # 可视化识别结果（将结果画到图片上）
    image = cv2.imread(image_path)
    boxes = [line[0] for line in result[0]]
    texts = [line[1][0] for line in result[0]]
    scores = [line[1][1] for line in result[0]]

    # 绘制结果图
    result_img = draw_ocr(image, boxes, texts, scores, font_path='/System/Library/Fonts/PingFang.ttc')
    cv2.imwrite(output_path, result_img)
    print(f"识别结果已保存至：{output_path}")

# 调用测试
if __name__ == '__main__':
    recognize_label_with_positions('testImage/motor_label_with_barcode.png', 'motor_label_result.png')
    recognize_label_with_positions('testImage/food_label_with_barcode.png', 'food_label_result.png')
