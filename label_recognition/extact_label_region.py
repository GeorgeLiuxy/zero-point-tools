import cv2
import numpy as np

def extract_label_region(image_path, output_path="cropped_label.png"):
    # 1. 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("图像读取失败，请检查路径是否正确")

    # 2. 转灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. 高斯模糊 + 边缘检测
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # 4. 闭操作增强边缘连接
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # 5. 查找轮廓
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未检测到任何标签轮廓")

    # 6. 获取最大轮廓区域
    label_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(label_contour)

    # 7. 裁剪标签区域
    cropped = image[y:y+h, x:x+w]
    cv2.imwrite(output_path, cropped)
    print(f"✅ 标签区域已保存为：{output_path}")

    return output_path


if __name__ == '__main__':
    extract_label_region("testImage/WechatIMG47.jpeg", "cropped_motor_label.png")
