import cv2
import numpy as np
import pywt  # 用于小波变换
from matplotlib import pyplot as plt

def embed_watermark(image_path, watermark_text, output_path, alpha=0.1, visible_opacity=0.3):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or the format is not supported.")

    # 获取图像的离散小波变换（DWT）
    coeffs2 = pywt.dwt2(image, 'haar')  # 使用 Haar 小波变换
    LL, (LH, HL, HH) = coeffs2

    # 创建一个水印图像（与 LL 尺寸相同）
    watermark = np.zeros_like(LL, dtype=np.float32)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(LL.shape) / 100  # 根据图像大小调整字体比例
    font_thickness = 1

    # 在水印图像的中心添加文本
    text_size, _ = cv2.getTextSize(watermark_text, font, font_scale, font_thickness)
    text_x = (watermark.shape[1] - text_size[0]) // 2
    text_y = (watermark.shape[0] + text_size[1]) // 2
    cv2.putText(watermark, watermark_text, (text_x, text_y), font, font_scale, (255), font_thickness, lineType=cv2.LINE_AA)

    # 将水印嵌入到图像的低频部分（LL）中
    LL_watermarked = LL + alpha * watermark  # 使用 alpha 控制水印强度

    # 重组带有水印的 DWT 图像
    watermarked_coeffs2 = (LL_watermarked, (LH, HL, HH))
    watermarked_image = pywt.idwt2(watermarked_coeffs2, 'haar')

    # 归一化图像到 8 位范围
    watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)

    # 在图像上直接添加可见水印
    overlay = watermarked_image.copy()
    cv2.putText(overlay, watermark_text, (text_x, text_y), font, font_scale, (255), font_thickness, lineType=cv2.LINE_AA)
    # 使用透明度叠加水印，使其更为柔和
    visible_watermarked_image = cv2.addWeighted(overlay, visible_opacity, watermarked_image, 1 - visible_opacity, 0)

    # 保存带有水印的图像
    cv2.imwrite(output_path, visible_watermarked_image)
    print(f"Watermarked image saved at: {output_path}")

    # 可视化水印图像和原图对比
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Watermarked Image")
    plt.imshow(visible_watermarked_image, cmap='gray')
    plt.axis("off")

    plt.show()

# 使用示例
image_path = "WX20241107-175237.png"
output_path = "watermarked_image13.jpg"
watermark_text = "麦极客在线"
embed_watermark(image_path, watermark_text, output_path)
