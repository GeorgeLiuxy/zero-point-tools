import cv2
import numpy as np

# 读取图像
image = cv2.imread('image.jpg')

# 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用 Canny 边缘检测
edges = cv2.Canny(gray, 100, 200)

# 进行闭运算，填补边缘中的空隙
kernel = np.ones((5, 5), np.uint8)
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# 查找轮廓
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建空白图像，用于绘制轮廓
output = np.zeros_like(image)

# 绘制轮廓
cv2.drawContours(output, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

# 显示结果
cv2.imshow('Extracted Region', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果
cv2.imwrite('extracted_region_canny.jpg', output)
