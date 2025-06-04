import math
import os
import cv2
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from scipy import interpolate

# --------- 中文字体查找和异常弹窗处理 ----------
def get_chinese_font():
    font_path = ""
    if os.name == 'nt':
        font_path = "C:/Windows/Fonts/simhei.ttf"
    elif os.name == 'posix':
        if os.path.exists("/System/Library/Fonts/STHeiti Light.ttc"):
            font_path = "/System/Library/Fonts/STHeiti Light.ttc"
        elif os.path.exists("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"):
            font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
        else:
            font_path = ""
    else:
        font_path = ""
    if not font_path or not os.path.exists(font_path):
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showwarning("字体文件未找到", "未找到中文字体，中文可能显示异常！")
        except Exception:
            print("未找到中文字体，且无法弹出窗口。中文可能显示异常！")
        return None
    else:
        return FontProperties(fname=font_path)

font_prop = get_chinese_font()
if font_prop:
    matplotlib.rcParams['font.family'] = font_prop.get_name()
matplotlib.rcParams['axes.unicode_minus'] = False

# 图像处理和平滑
def process_image(img_path, window_length=15, polyorder=2):  # Increased window_length for better fitting
    img = cv2.imread(img_path, 0)
    rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # 使用中值滤波进行降噪
    blurred_img = cv2.medianBlur(rotated_img, 5)
    edges = cv2.Canny(blurred_img, 100, 200)
    adaptive_thresh = cv2.adaptiveThreshold(edges.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((5, 5), np.uint8)
    dilated_img = cv2.dilate(adaptive_thresh, kernel, iterations=1)
    eroded_img = cv2.erode(dilated_img, kernel, iterations=1)

    contours, _ = cv2.findContours(eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    contour = contour[:, 0, :]

    # 用立方插值提高平滑精度
    t = np.linspace(0, 1, len(contour))  # 原始点的参数化
    t_new = np.linspace(0, 1, 500)  # 新的参数范围，增加点数
    interpolator = interpolate.CubicSpline(t, contour)
    smoothed_contour = interpolator(t_new)
    return rotated_img, smoothed_contour

# 去除相邻不规则的锯齿谷或锯齿尖，并合并距离过近的点
def remove_nearby_irregular_zigzags(peaks, threshold_distance=15):
    optimized_peaks = []
    previous_peak = peaks[0]  # 初始化第一个点

    for i in range(1, len(peaks)):
        current_peak = peaks[i]
        dist = calculate_distance(previous_peak, current_peak)

        if dist < threshold_distance:  # 如果相邻点的距离小于阈值，就合并
            continue
        else:
            optimized_peaks.append(previous_peak)
            previous_peak = current_peak

    optimized_peaks.append(previous_peak)
    return optimized_peaks

# 获取轮廓图的左右两侧极点
def get_edge_points(contour):
    leftmost = min(contour, key=lambda x: x[0])  # 获取最左边的点
    rightmost = max(contour, key=lambda x: x[0])  # 获取最右边的点
    return leftmost, rightmost

# 峰谷识别
def identify_peaks_and_valleys(part, is_upper_part=True):
    peaks = []
    diff = np.diff(part[:, 1])
    for i in range(1, len(diff) - 1):
        if is_upper_part:
            if diff[i - 1] < 0 and diff[i + 1] > 0:
                peaks.append([part[i, 0], part[i, 1]])  # 锯齿谷
        else:
            if diff[i - 1] > 0 and diff[i + 1] < 0:
                peaks.append([part[i, 0], part[i, 1]])  # 锯齿尖
    return peaks

# 计算两点之间的距离
def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# 配对点（按距离最小配对）
def pair_points_by_distance(upper_peaks, lower_peaks, max_distance=50):
    paired_points = []
    lower_used = [False] * len(lower_peaks)

    for upper_point in upper_peaks:
        closest_point = None
        min_distance = float('inf')
        for i, lower_point in enumerate(lower_peaks):
            if not lower_used[i]:
                dist = calculate_distance(upper_point, lower_point)
                if dist < min_distance and dist <= max_distance:  # 限制配对点的最大距离
                    min_distance = dist
                    closest_point = lower_point
                    closest_index = i

        if closest_point is not None:
            paired_points.append((upper_point, closest_point))
            lower_used[closest_index] = True  # 标记该点已被配对

    return paired_points

# 标注同侧相邻点的距离
def annotate_distances_on_same_side(points, color='blue'):
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        dist = calculate_distance(p1, p2)
        midpoint = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
        # 在相邻点之间标注距离
        plt.text(midpoint[0], midpoint[1], f'{dist:.2f}', color=color, fontsize=8, ha='center', va='center', fontweight='bold')

# 展示配对结果（带有连线的标号和长度）
def plot_paired_points(rotated_img, smoothed_contour, paired_points, show_smooth_curve=True, alpha=0.5):
    plt.figure(figsize=(10, 10), dpi=150)  # 增加图像大小和分辨率
    plt.imshow(rotated_img, cmap='gray', alpha=alpha)  # 设置透明度 0.5

    if show_smooth_curve:
        plt.plot(smoothed_contour[:, 0], smoothed_contour[:, 1], 'b-', alpha=alpha, label='优化后的轮廓')

    # 为每个配对分配一个颜色
    colors = plt.cm.jet(np.linspace(0, 1, len(paired_points)))  # 生成一组颜色
    for i, (upper_point, lower_point) in enumerate(paired_points):
        plt.plot([upper_point[0], lower_point[0]], [upper_point[1], lower_point[1]],
                 color=colors[i], linewidth=1)  # 增加线条宽度

    # 标注距离
    annotate_distances_on_same_side([point[0] for point in paired_points], 'green')
    annotate_distances_on_same_side([point[1] for point in paired_points], 'red')

    # 显示最左和最右的点
    leftmost, rightmost = get_edge_points(smoothed_contour)
    plt.scatter([leftmost[0], rightmost[0]], [leftmost[1], rightmost[1]], c='red', label='两侧极点', marker='o')

    # 显示图例并关闭坐标轴
    plt.title('配对的锯齿点及其连线', fontsize=14)
    plt.axis('off')
    plt.show()

# 主方法
def main(img_path, show_smooth_curve, alpha=0.5, window_length=9, polyorder=3, angle_threshold=45):
    rotated_img, smoothed_contour = process_image(img_path, window_length, polyorder)

    height = rotated_img.shape[0]
    middle_y = height // 2

    upper_part = smoothed_contour[smoothed_contour[:, 1] < middle_y]
    lower_part = smoothed_contour[smoothed_contour[:, 1] >= middle_y]

    upper_peaks = identify_peaks_and_valleys(upper_part, is_upper_part=True)
    lower_peaks = identify_peaks_and_valleys(lower_part, is_upper_part=False)

    optimized_upper_peaks = remove_nearby_irregular_zigzags(upper_peaks)
    optimized_lower_peaks = remove_nearby_irregular_zigzags(lower_peaks)

    # 配对上下锯齿点并过滤不符合条件的配对
    paired_points = pair_points_by_distance(optimized_upper_peaks, optimized_lower_peaks)

    # 展示配对的结果
    plot_paired_points(rotated_img, smoothed_contour, paired_points, show_smooth_curve, alpha)

if __name__ == '__main__':
    # 调用主方法，设置合适的角度阈值
    main('images/311747292980_.pic.jpg', False, alpha=0.5, window_length=9, polyorder=2, angle_threshold=20)
