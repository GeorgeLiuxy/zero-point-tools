import cv2
import numpy as np
import matplotlib.pyplot as plt

# 图像增强（像素提升）
def enhance_image(img_path):
    # 读取原图
    img = cv2.imread(img_path, 0)

    # 直方图均衡化
    hist_eq_img = cv2.equalizeHist(img)

    # 锐化滤波器
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # 锐化内核
    sharpened_img = cv2.filter2D(hist_eq_img, -1, kernel)

    return sharpened_img

# 自动生成掩码并进行图像修复（inpainting）
def auto_inpainting(img_path):
    # 读取原图
    img = cv2.imread(img_path, 0)  # 灰度图

    # Canny边缘检测
    edges = cv2.Canny(img, 100, 200)

    # 形态学操作，膨胀和腐蚀
    kernel = np.ones((5, 5), np.uint8)
    dilated_img = cv2.dilate(edges, kernel, iterations=1)
    eroded_img = cv2.erode(dilated_img, kernel, iterations=1)

    # 掩码是从边缘图像生成的，白色区域为需要修复的部分
    mask = cv2.bitwise_not(eroded_img)  # 将边缘区域反转为掩码

    # 使用inpainting修复图像
    inpainted_img = cv2.inpaint(cv2.imread(img_path), mask, 3, cv2.INPAINT_TELEA)

    return inpainted_img, mask

# 区域生长法补全缺失部分
def region_growing(img, seed_point, threshold=50, max_growth=500):
    h, w = img.shape
    region = np.zeros_like(img, dtype=np.uint8)
    region[seed_point[1], seed_point[0]] = 255  # 标记种子点
    growth_points = [seed_point]

    while growth_points and max_growth > 0:
        point = growth_points.pop()
        x, y = point
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]  # 上下左右邻居
        for nx, ny in neighbors:
            if 0 <= nx < w and 0 <= ny < h and region[ny, nx] == 0:  # 检查边界
                if abs(int(img[y, x]) - int(img[ny, nx])) < threshold:  # 判断邻居像素差异
                    region[ny, nx] = 255  # 填充区域
                    growth_points.append((nx, ny))  # 继续扩展
                    max_growth -= 1
    return region

# 主函数，整合增强、修复和区域生长
def main(img_path, seed_point):
    # 1. 图像增强（锐化 + 直方图均衡化）
    enhanced_img = enhance_image(img_path)

    # 2. 自动生成掩码并进行图像修复（inpainting）
    inpainted_img, mask = auto_inpainting(img_path)

    # 3. 区域生长补全
    img = cv2.imread(img_path, 0)  # 读取原图灰度图
    filled_region = region_growing(img, seed_point)

    # 显示结果
    plt.figure(figsize=(10, 8))

    # 显示增强后的图像
    plt.subplot(131)
    plt.imshow(enhanced_img, cmap='gray')
    plt.title("Enhanced Image")

    # 显示图像修复结果
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB))
    plt.title("Inpainted Image")

    # 显示区域生长补全结果
    plt.subplot(133)
    plt.imshow(filled_region, cmap='gray')
    plt.title("Region Growing Fill")

    plt.tight_layout()
    plt.show()

# 测试用例
img_path = "images/311747292980_.pic.jpg"  # 图像路径
seed_point = (100, 100)  # 区域生长的种子点位置

# 调用主函数
main(img_path, seed_point)
