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


# 图像处理和平滑函数
def process_image(img_path, window_length=15, polyorder=2):  # Increased window_length for better fitting
    img = cv2.imread(img_path, 0)
    rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # 高斯模糊去噪
    blurred_img = cv2.GaussianBlur(rotated_img, (1, 1), 0)
    # Canny边缘检测（阈值可以调节）
    edges = cv2.Canny(blurred_img, 1, 3)  # 调整阈值
    # 自适应阈值
    adaptive_thresh = cv2.adaptiveThreshold(edges.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)
    # 形态学操作：膨胀 + 腐蚀
    kernel = np.ones((1, 1), np.uint8)
    dilated_img = cv2.dilate(adaptive_thresh, kernel, iterations=1)
    eroded_img = cv2.erode(dilated_img, kernel, iterations=1)
    # 提取轮廓
    contours, _ = cv2.findContours(eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    contour = contour[:, 0, :]
    # 用B样条插值提高平滑精度
    t = np.linspace(0, 1, len(contour))  # 原始点的参数化
    t_new = np.linspace(0, 1, 100000)  # 增加点数（更高的分辨率）
    interpolator = interpolate.CubicSpline(t, contour)
    smoothed_contour = interpolator(t_new)

    return rotated_img, smoothed_contour

def pair_with_constraints(upper_peaks, lower_peaks, max_distance=50):
    paired_points = []
    for upper_point in upper_peaks:
        for lower_point in lower_peaks:
            dist = calculate_distance(upper_point, lower_point)
            if dist <= max_distance:  # 限制配对点的最大距离
                paired_points.append((upper_point, lower_point))
                break
    return paired_points


# 使用距离最小的原则进行配对
def pair_points_by_distance(upper_peaks, lower_peaks):
    paired_points = []
    lower_used = [False] * len(lower_peaks)

    for upper_point in upper_peaks:
        closest_point = None
        min_distance = float('inf')
        for i, lower_point in enumerate(lower_peaks):
            if not lower_used[i]:
                dist = calculate_distance(upper_point, lower_point)
                if dist < min_distance:
                    min_distance = dist
                    closest_point = lower_point
                    closest_index = i

        if closest_point is not None:
            paired_points.append((upper_point, closest_point))
            lower_used[closest_index] = True  # 标记该点已被配对

    return paired_points


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


# 计算两点之间与垂直线的夹角
def calculate_angle(p1, p2):
    # 计算两点之间的连线角度（相对于水平线）
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    angle = math.atan2(delta_y, delta_x)  # 计算与水平线的夹角
    angle_deg = np.degrees(angle)  # 转换为度数

    # 计算与垂直线的夹角，垂直线的角度是90° - 水平线的角度
    angle_with_vertical = abs(90 - abs(angle_deg))

    # 返回绝对值确保夹角为正
    return angle_with_vertical


# 通过平滑曲线，配对上下部分的锯齿点并计算夹角
def pair_and_filter_peaks(upper_peaks, lower_peaks, angle_threshold=45):
    paired_points = []
    lower_used = [False] * len(lower_peaks)  # 标记下半部分的点是否已被配对

    # 遍历上半部分的每个锯齿点
    for upper_point in upper_peaks:
        for lower_index in range(len(lower_peaks)):
            if lower_used[lower_index]:  # 如果下半部分的点已经配对过，跳过
                continue

            lower_point = lower_peaks[lower_index]
            angle = calculate_angle(upper_point, lower_point)

            if angle <= angle_threshold:  # 如果夹角符合要求，进行配对
                paired_points.append((upper_point, lower_point))
                lower_used[lower_index] = True  # 标记下半部分的点已配对
                break  # 跳出内层循环，继续处理下一个上半部分的点

    return paired_points


# 计算两点之间的距离
def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


# 同一侧相邻两个点的距离标注
def annotate_distances(peaks, color):
    if isinstance(peaks, list):  # Check if peaks is a list
        for i in range(1, len(peaks)):
            p1 = peaks[i - 1]
            p2 = peaks[i]
            dist = calculate_distance(p1, p2)
            midpoint = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
            # 在相邻点之间标注距离
            plt.text(midpoint[0], midpoint[1], f'{dist:.2f}', color=color, fontsize=8, ha='center', va='center', fontweight='bold')
    else:
        print(f"Warning: peaks should be a list, but received {type(peaks)}")


# 标注同侧相邻点的距离
def annotate_distances_on_same_side(points, color='blue'):
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        dist = calculate_distance(p1, p2)
        midpoint = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
        # 在相邻点之间标注距离
        plt.text(midpoint[0], midpoint[1], f'{dist:.2f}', color=color, fontsize=8, ha='center', va='center', fontweight='bold')


# 计算两点之间的向量
def vector_from_points(p1, p2):
    return (p2[0] - p1[0], p2[1] - p1[1])

# 计算两个向量之间的夹角
def calculate_angle_between_vectors(v1, v2):
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)

    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)

    # 防止因精度误差导致的acos输入超出范围
    cos_theta = max(-1.0, min(1.0, cos_theta))

    angle_rad = math.acos(cos_theta)  # 返回弧度
    angle_deg = math.degrees(angle_rad)  # 转换为角度
    return angle_deg


# 计算由三点A, B, C所形成的角度
def calculate_angle_with_extremes(start_point, end_point, most_point):
    # 计算向量 AB 和 BC
    AB = vector_from_points(start_point, most_point)
    CB = vector_from_points(end_point, most_point)
    # 计算AB和BC之间的夹角
    angle = calculate_angle_between_vectors(AB, CB)
    return angle


# 展示配对结果（带有连线的标号和长度）
def plot_paired_points(rotated_img, smoothed_contour, paired_points, show_smooth_curve=True, alpha=0.5):
    # 创建图像并绘制所有内容
    plt.figure(figsize=(10, 10), dpi=150)  # 增加图像大小和分辨率
    plt.imshow(rotated_img, cmap='gray', alpha=alpha)  # 设置透明度 0.5

    if show_smooth_curve:
        plt.plot(smoothed_contour[:, 0], smoothed_contour[:, 1], 'b-', alpha=alpha, label='优化后的轮廓')

    # 为每个配对分配一个颜色
    colors = plt.cm.jet(np.linspace(0, 1, len(paired_points)))  # 生成一组颜色

    table_df = calc_line_length(colors, paired_points)
    calc_neighbor_distance(table_df)

    # 获取并记录左右两侧的极点
    leftmost, rightmost = get_edge_points(smoothed_contour)
    #  计算每条连线的夹角
    left_most_line_angles = []
    right_most_line_angles = []
    # 遍历到倒数第二个点（避免越界）
    for i in range(len(table_df) - 1):
        # 当前点和下一个点的连线编号
        current_line_number = table_df.iloc[i, 0]  # 使用.iloc[i, 0]访问当前连线编号
        start_point = table_df.iloc[i, 1]
        end_point = table_df.iloc[i, 2]
        #  计算当前连线与左侧极点角度
        left_line_angle = calculate_angle_with_extremes(start_point, end_point, leftmost)
        left_most_line_angles.append([i + 1, current_line_number, leftmost, left_line_angle])
        # 创建一个DataFrame保存相邻起点之间的距离、连线编号以及起点坐标
        left_line_angle_df = pd.DataFrame(left_most_line_angles, columns=['序号', '连线编号', '底部极点坐标', '角度'])
        # 结果保存到文件中，数值类型转换成字符串
        left_line_angle_df.to_csv('left_most_line_angles_info.csv', index=False)
        #  计算当前连线与右侧极点角度
        right_line_angle = calculate_angle_with_extremes(start_point, end_point, rightmost)
        right_most_line_angles.append([i + 1, current_line_number, rightmost, right_line_angle])
        # 创建一个DataFrame保存相邻起点之间的距离、连线编号以及起点坐标
        right_line_angle_df = pd.DataFrame(right_most_line_angles, columns=['序号', '连线编号', '顶部极点坐标', '角度'])
        # 结果保存到文件中，数值类型转换成字符串
        right_line_angle_df.to_csv('right_most_line_angles_info.csv', index=False)

    # 显示最左和最右的点
    plt.scatter([leftmost[0], rightmost[0]], [leftmost[1], rightmost[1]], c='red', label='两侧极点', marker='o')

    # 为图例添加唯一标签，避免重复
    plt.title('左右部分配对的锯齿点及其连线', fontsize=14)  # 增加标题字体大小
    # plt.legend(fontsize=12, loc='best')  # 增加图例字体大小
    plt.axis('off')
    # plt.show()
    # 保存当前绘制的图像，并确保高分辨率
    file_path = "original_image.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0.1)  # 使用300 DPI

    # 读取保存的图像
    img = mpimg.imread(file_path)

    # 逆时针旋转图像90度
    rotated_img = np.rot90(img)

    # 保存旋转后的图像
    rotated_file_path = "rotated_image.png"
    plt.imsave(rotated_file_path, rotated_img)

    # # 显示旋转后的图像
    # plt.imshow(rotated_img)
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()


def calc_line_length(colors, paired_points):
    # 初始化表格数据
    table_data = []
    # 在每一条连线上添加标号和长度
    for i, (upper_point, lower_point) in enumerate(paired_points):
        # 绘制连接的线，增加线条宽度
        plt.plot([upper_point[0], lower_point[0]], [upper_point[1], lower_point[1]],
                 color=colors[i], linewidth=0.5)  # 仅为第一个连线添加标签

        # 计算长度
        length = calculate_distance(upper_point, lower_point)

        # 记录数据到表格中
        table_data.append([i + 1, upper_point, lower_point, length])

        # 在起始点标记编号，纵向错层排列
        offset_y = 10  # 设置垂直偏移量
        if i % 2 == 0:
            # 偶数编号，标记向上
            offset_y = offset_y
        else:
            # 奇数编号，标记向下
            offset_y = -offset_y

        plt.annotate(f'{i + 1}', xy=(upper_point[0], upper_point[1]),
                     xytext=(upper_point[0], upper_point[1] + offset_y),  # 使用计算的偏移量
                     fontsize=8, ha='center', va='center', fontweight='bold', color='black')
    # 使用pandas表格显示编号和长度
    table_df = pd.DataFrame(table_data, columns=['编号', '起始点', '结束点', '长度'])
    # 结果保存到文件中 数值类型转换成字符串
    table_df.to_csv('pair_length_info.csv', index=False)
    return table_df


def calc_neighbor_distance(table_df):
    # 根据 table_df 连线起点进行排序，然后将相邻起点的距离计算出来并保存到文件中
    # 计算相邻起点之间的距离，并保存到文件
    table_df = table_df.sort_values(by='起始点')
    # 初始化保存相邻起点之间的距离的列表
    distances = []
    # 遍历到倒数第二个点（避免越界）
    for i in range(len(table_df) - 1):
        # 当前点和下一个点的连线编号
        current_line_number = table_df.iloc[i, 0]  # 使用.iloc[i, 0]访问当前连线编号
        next_line_number = table_df.iloc[i + 1, 0]  # 使用.iloc[i + 1, 0]访问下一个连线编号
        prev_point = table_df.iloc[i, 1]  # 当前起始点
        next_point = table_df.iloc[i + 1, 1]  # 下一个起始点
        distance = calculate_distance(prev_point, next_point)
        # 将数据添加到distances列表中
        distances.append([i + 1, current_line_number, next_line_number, distance])  # 编号从1开始
    # 创建一个DataFrame保存相邻起点之间的距离、连线编号以及起点坐标
    distance_df = pd.DataFrame(distances, columns=['序号', '当前连线编号', '相邻下一个点连线编号', '起点相邻距离'])
    # 结果保存到文件中，数值类型转换成字符串
    distance_df.to_csv('neighbor_distance_info.csv', index=False)


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
    paired_points = pair_and_filter_peaks(optimized_upper_peaks, optimized_lower_peaks, angle_threshold)
    # 对配对点按 x 坐标排序，避免交叉
    paired_points = sorted(paired_points, key=lambda pair: (pair[0][0], pair[1][0]))

    # rotated_img：原图（旋转后的图像），显示为背景。
    # smoothed_contour：平滑后的轮廓，显示为平滑曲线。
    # paired_points：经过角度筛选后的配对点。
    # show_smooth_curve：是否显示平滑曲线。
    # alpha：透明度，控制各元素的透明度。
    plot_paired_points(rotated_img, smoothed_contour, paired_points, show_smooth_curve, alpha)


if __name__ == '__main__':
    # 调用主方法，设置合适的角度阈值
    main('images/11.jpg', True, alpha=0.5, window_length=9, polyorder=2, angle_threshold=20)
    # main('images/1.jpeg', False, alpha=0.5, window_length=9, polyorder=2, angle_threshold=20)
