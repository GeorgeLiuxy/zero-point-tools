import numpy as np
from scipy.interpolate import splprep, splev

def b_spline_interpolation(smoothpath_list):

    # 定义要插入的点数（在每对相邻点之间）
    num_new_points_between = 3

    # 创建一个空的列表来存储扩展后的路径
    smooth_path_expanded = []

    # 遍历 smooth_path 中的相邻点对
    for i in range(len(smoothpath_list) - 1):
        point1 = smoothpath_list[i]
        point2 = smoothpath_list[i + 1]

        # 计算两点之间的差异
        x_diff = point2[0] - point1[0]
        y_diff = point2[1] - point1[1]

        # 将第一个点添加到扩展后的路径中
        smooth_path_expanded.append(point1)

        # 计算并添加插值点
        for j in range(num_new_points_between):
            t = (j + 1) / (num_new_points_between + 1)  # 计算插值参数 t
            new_point = [point1[0] + t * x_diff, point1[1] + t * y_diff]
            smooth_path_expanded.append(new_point)

    # 添加最后一个点到扩展后的路径中（如果没有在最后一对点之后添加额外点的话）
    smooth_path_expanded.append(smoothpath_list[-1])
    smooth_path_expanded = np.array(smooth_path_expanded)
    # print(smooth_path_expanded)
    # print(smooth_path_expanded.shape[0])

    # 参数化以准备进行B样条拟合
    # s是平滑因子，越大越平滑，但也可能导致曲线偏离原始点更远
    s = 1  # 可以尝试不同的值，比如0.1, 1.0等
    tck, u = splprep(smooth_path_expanded.T, s=s)

    # 生成新的平滑路径点，这里使用了100个点来绘制平滑曲线
    new_points = splev(np.linspace(0, 1, 100), tck)
    result_path = np.vstack(new_points).T  # 转置以匹配原始路径的格式

    return result_path