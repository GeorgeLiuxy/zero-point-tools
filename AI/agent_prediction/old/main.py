import matplotlib
matplotlib.use('TkAgg')  # 确保使用支持交互的后端
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from Correspondence import is_link_active
from NLoS_probability import count_obstacles
from global_path_generator import find_globalpath
from gridsca import dasca_f
# from interruption_rate import transitive_closure
from spline_utilities import b_spline_interpolation
import collections
from CostFunction import myCost

# 定义栅格地图
grid_size = 100
grid_resolution = 1
grid_map = np.zeros((grid_size, grid_size))
costFunction = myCost

# 障碍物
obstacles = [
    ((25, 45), 4),
    ((35, 10), 2),
    ((35, 55), 2),
    ((40, 62), 2),
    ((55, 50), 3),
    ((75, 50), 4),
    ((70, 37), 3),
    ((75, 62), 3),
    ((55, 55), 2),
    ((60, 75), 2)
]

for center, radius in obstacles:
    cx, cy = center
    x_min = max(0, int(cx - radius))
    x_max = min(grid_size, int(cx + radius))
    y_min = max(0, int(cy - radius))
    y_max = min(grid_size, int(cy + radius))
    # 使用圆形遮罩更精确地绘制圆形障碍物
    Y, X = np.ogrid[:grid_size, :grid_size]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = dist_from_center <= radius
    grid_map[mask] = 1.0

# 机器人的起点和终点
totalRobot = 8
robotRadius = np.ones(totalRobot)/2  # 机器人半径
robotStart = np.array([
    [2.0, 52.0],
    [2.0, 55.0],
    [2.0, 58.0],
    [2.0, 60.0],
    [5.0, 52.0],
    [5.0, 55.0],
    [5.0, 58.0],
    [5.0, 60.0]
])
robotGoal = np.array([
    [82.0, 53.0],
    [60.0, 50.0],
    [83.0, 70.0],
    [70.0, 52.0],
    [65.0, 65.0],
    [82.0, 50.0],
    [70.0, 60.0],
    [85.0, 30.0]
])
colors = ['#1AF6F9','#3BB50B','#BA0652','#FFA3FE','#EBBC1C','#9E8568','#9EF9BE','#73FE5C']

# 动态障碍物的起点和终点
totalDynamic = 5
dynamicStart = np.array([
    [20.0, 130.0],
    [40.0, 90.0],
    [65.0, 130.0],
    [80.0, 85.0],
    [120.0, 50.0]
])/2
dynamicGoal = np.array([
    [30.0, 75.0],
    [60, 130.0],
    [80.0, 70.0],
    [100.0, 155.0],
    [140.0, 100.0]
])/2
dynamicRadius = np.array([1.5, 1.5, 1.5, 1.5, 1.5])/2
dynamicVelocity = [0.5, 0.5, 0.6, 0.3, 0.4]
dynamicVelocity = [x / 2 for x in dynamicVelocity]
axisParameter = 100

# 初始化机器人位置
robotOld = robotStart.copy()
robot_old_x = robotOld[:, 0]
robot_old_y = robotOld[:, 1]

# 初始化动态障碍物位置
dynamicOld = dynamicStart.copy()

# 初始化绘图
plt.ion()  # 启用交互模式
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect('equal')
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)

# 使用imshow高效绘制障碍物
ax.imshow(grid_map, cmap='gray', origin='lower', extent=[0, grid_size, 0, grid_size])

# 绘制机器人及其目标
for i in range(totalRobot):
    circle = plt.Circle((robotStart[i, 0], robotStart[i, 1]), robotRadius[i], color=[0.5, 0.7, 0.8], fill=True, alpha=0.5)
    ax.add_patch(circle)

    ax.plot(robotGoal[i, 0], robotGoal[i, 1], 'x', markersize=5, color=colors[i])
    ax.plot([robotStart[i, 0], robotGoal[i, 0]], [robotStart[i, 1], robotGoal[i, 1]], '--', linewidth=0.5, color=(0.75, 0.75, 0.75, 0.5))

# 绘制动态障碍物
dynamic_patches = []
for i in range(totalDynamic):
    circle = plt.Circle((dynamicStart[i, 0], dynamicStart[i, 1]), dynamicRadius[i], color=[0.1, 0.2, 0.9], fill=True, alpha=0.5)
    ax.add_patch(circle)
    dynamic_patches.append(circle)
    ax.plot(dynamicGoal[i, 0], dynamicGoal[i, 1], 'x', markerfacecolor='b', markeredgecolor='b', markersize=5, linewidth=2)
    ax.plot([dynamicStart[i, 0], dynamicGoal[i, 0]], [dynamicStart[i, 1], dynamicGoal[i, 1]], '--b')

# 初始化机器人距离
distanceToGoal = np.linalg.norm(robotOld - robotGoal, axis=1)

# 设置停止条件
stoppingCriteria = max(distanceToGoal)
robotNew = robotOld.copy()
dynamicNew = np.zeros_like(dynamicOld)

# 用于存储机器人连线的Line2D对象列表
robot_lines = []
A_starlines = []

# 控制参数
maximumFitness = 200
populationSize = 50
dimension = 2
maximumVelocity = 1
minimumVelocity = 0.5
minimumTheta = 0
maximumTheta = 6.2832
step = 1

# 初始化机器人路径记录为二维数组
robotPath_x = robotOld[:, 0].reshape(1, -1)  # 形状 (1, 8)
robotPath_y = robotOld[:, 1].reshape(1, -1)  # 形状 (1, 8)

# 初始化动态障碍物路径记录为二维数组
dynamicPath_x = dynamicOld[:, 0].reshape(1, -1)  # 形状 (1, 5)
dynamicPath_y = dynamicOld[:, 1].reshape(1, -1)  # 形状 (1, 5)

# 初始化机器人的历史轨迹队列，队列长度可以根据需要调整
history_length = 10
robot_historys = [collections.deque(maxlen=history_length) for _ in range(totalRobot)]

# 定义初始距离
initial_distances = np.linalg.norm(robotOld - robotGoal, axis=1)

# 定义局部地图
localgridmap = np.zeros_like(grid_map)

# 主循环
while stoppingCriteria > 1:
    grid_robotmaps = []
    A_starpaths = []
    dynamic_circles = []
    robot_circles = []
    bestSols = []
    step += 1

    # 更新动态障碍物信息
    dynamicDistanceToGoal = np.linalg.norm(dynamicOld - dynamicGoal, axis=1)
    slope = np.zeros(totalDynamic)
    for i in range(totalDynamic):
        if dynamicDistanceToGoal[i] > 1:
            slope[i] = math.atan2(dynamicGoal[i, 1] - dynamicOld[i, 1], dynamicGoal[i, 0] - dynamicOld[i, 0])
            dynamicNew[i, 0] = dynamicOld[i, 0] + dynamicVelocity[i] * math.cos(slope[i])
            dynamicNew[i, 1] = dynamicOld[i, 1] + dynamicVelocity[i] * math.sin(slope[i])

            print(f"动态障碍物 {i} 新位置: {dynamicNew[i]}")

        # 可视化动态障碍物
        circle = patches.Circle((dynamicNew[i, 0], dynamicNew[i, 1]), dynamicRadius[i], color=(0.1, 0.2, 0.9), fill=True)
        ax.add_patch(circle)
        dynamic_circles.append(circle)

    # 更新栅格图以反映动态障碍物所占的栅格
    grid_dynamicmap = np.zeros((grid_size, grid_size))
    for index, pos in enumerate(dynamicNew):
        x, y = pos
        x_min = max(0, int(x - 0.75))
        x_max = min(grid_size, int(x + 0.75))
        y_min = max(0, int(y - 0.75))
        y_max = min(grid_size, int(y + 0.75))
        # 使用圆形遮罩更精确地绘制圆形动态障碍物
        Y, X = np.ogrid[:grid_size, :grid_size]
        dist_from_center = np.sqrt((X - x)**2 + (Y - y)**2)
        mask = dist_from_center <= 0.75
        grid_dynamicmap[mask] = 1

    # 更新全局图
    combined_grid = np.logical_or(grid_map, grid_dynamicmap).astype(int)

    # 更新栅格图以反映机器人所占的栅格
    for i in range(totalRobot):
        grid_robotmap = np.zeros((grid_size, grid_size))
        for j in range(totalRobot):
            if i != j:
                x, y = robotOld[j]
                x_min = max(0, int(x - 0.5))
                x_max = min(grid_size, int(x + 0.5))
                y_min = max(0, int(y - 0.5))
                y_max = min(grid_size, int(y + 0.5))
                grid_robotmap[y_min:y_max + 1, x_min:x_max + 1] = 1
        grid_robotmaps.append(grid_robotmap)

    # 提取当前每个机器人观测图
    for i in range(totalRobot):
        x, y = int(robotNew[i, 0]), int(robotNew[i, 1])
        x_min, x_max = max(0, x - 5), min(grid_size, x + 6)
        y_min, y_max = max(0, y - 5), min(grid_size, y + 6)
        # 更新局部地图：将全局地图中对应范围的信息复制到局部地图中
        localgridmap[y_min:y_max, x_min:x_max] = combined_grid[y_min:y_max, x_min:x_max]

    # 合并更新机器人观测矩阵
    for i in range(totalRobot):
        combined_grid_robot = np.logical_or(grid_robotmaps[i], localgridmap).astype(int)
        grid_robotmaps[i] = combined_grid_robot

    for i in range(totalRobot):
        # 使用A-star算法计算全局路径
        A_starpath = find_globalpath(robotOld[i], robotGoal[i], grid_robotmaps[i])

        # 计算到目标的距离
        distanceToGoal[i] = np.linalg.norm(robotOld[i] - robotGoal[i])

        if len(A_starpath) >= 2:
            # 计算一个基于距离的比例值，使用指数函数
            proportion = math.exp(-distanceToGoal[i] / initial_distances[i] * 4)
            # 使用比例值来计算适应因子
            adaptfactor = 5 + (100 - 5) * proportion
            A_starpath = b_spline_interpolation(A_starpath)

            # 计算与第一个元素的差值
            differences = np.linalg.norm(A_starpath - A_starpath[0], axis=1)

            # 找出差值大于3的第一个元素的下标
            index = np.argmax(differences > 3) if np.any(differences > 3) else -1

            localrobotGoal = A_starpath[index] if index != -1 else robotGoal[i]
        else:
            A_starpath = [robotOld[i].tolist()]
            localrobotGoal = robotGoal[i]

        A_starpaths.append(A_starpath)

        # 调用优化算法
        bestSol, bestCost = dasca_f(
            maximumFitness, populationSize, dimension, maximumVelocity, minimumVelocity,
            minimumTheta, maximumTheta, robotNew, dynamicNew, totalRobot, robotRadius,
            localrobotGoal, grid_map, robotGoal[i],
            totalDynamic, dynamicRadius,
            costFunction, robotOld[i], i
        )
        print(f"机器人 {i} 的最佳解决方案: 速度={bestSol[0]}, 角度={bestSol[1]}, 适应度={bestCost}")
        # bestSols.extend(bestSol)  # 移除这一行，因为我们不再使用 bestSols

        # 如果距离大于1，则更新机器人的位置
        if distanceToGoal[i] > 1:
            # 插值运算，更新机器人的转向角信息
            bestSol[1] = np.interp(bestSol[1], [minimumVelocity, maximumVelocity], [minimumTheta, maximumTheta])
            robotNew[i, 0] = robotOld[i, 0] + bestSol[0] * math.cos(bestSol[1])
            robotNew[i, 1] = robotOld[i, 1] + bestSol[0] * math.sin(bestSol[1])
            print(f"机器人 {i} 新位置: {robotNew[i]}")

        # 更新A_starpath的起点为机器人的新位置
        if len(A_starpaths[i]) > 0:
            A_starpaths[i][0][0] = robotNew[i, 0]
            A_starpaths[i][0][1] = robotNew[i, 1]

    # 更新机器人的历史动作轨迹队列
    for i in range(totalRobot):
        robot_historys[i].append((bestSol[0], bestSol[1]))

    # 可视化
    # 机器人位置与全局路径可视化更新
    for i in range(totalRobot):
        circle = patches.Circle((robotNew[i, 0], robotNew[i, 1]), robotRadius[i], color=colors[i], fill=True)
        ax.add_patch(circle)
        robot_circles.append(circle)
        # 实时全局路径更新
        A_starline, = ax.plot(
            [p[0] for p in A_starpaths[i]],
            [p[1] for p in A_starpaths[i]],
            linestyle='-.',
            linewidth=0.5,
            color=colors[i]
        )
        A_starlines.append(A_starline)

    # 计算机器人间的通信链路
    num_robots = len(robotNew)
    robot_distances = np.zeros((num_robots, num_robots))
    robot_links = np.zeros((num_robots, num_robots))
    robot_snrs = np.zeros((num_robots, num_robots))
    for i in range(len(robotNew)):
        for j in range(len(robotNew)):
            if i == j:
                robot_distances[i, j] = np.inf
                p_los = 1
            else:
                robot_distances[i, j] = np.linalg.norm(robotNew[i] - robotNew[j])
                p_los = count_obstacles(robotNew[i], robotNew[j], grid_map, grid_size) / 10
            robot_links[i, j], capacity, los, robot_snrs[i, j] = is_link_active(robot_distances[i, j], p_los)

    # 初始化一个集合来跟踪已经绘制的线条
    drawn_lines = set()
    # 将每个信噪比最高的临近机器人索引放到nearest_indices中
    # 每个机器人最大与三个机器人直接通信
    for i in range(len(robotNew)):
        # 排序信噪比并选择前三个
        nearest_indices = np.argsort(robot_snrs[i])[::-1][:3]
        for j in nearest_indices:
            if robot_links[i, j] and i != j:
                # 创建一个表示线条的元组，用于检查是否已绘制过这条线
                line_tuple = (i, j) if i < j else (j, i)
                # 如果这条线还没有绘制过，则绘制它，并将其添加到已绘制线条的集合中
                if line_tuple not in drawn_lines:
                    line, = ax.plot(
                        [robotNew[i, 0], robotNew[j, 0]],
                        [robotNew[i, 1], robotNew[j, 1]],
                        linestyle='--',
                        linewidth=0.25,
                        color='#6684e5'
                    )
                    robot_lines.append(line)
                    drawn_lines.add(line_tuple)

    # 更新机器人和障碍物的位置和路径
    robotOld = robotNew.copy()

    robotPath_x = np.vstack((robotPath_x, robotNew[:, 0].copy().reshape(1, -1)))
    robotPath_y = np.vstack((robotPath_y, robotNew[:, 1].copy().reshape(1, -1)))

    dynamicOld = dynamicNew.copy()

    dynamicPath_x = np.vstack((dynamicPath_x, dynamicNew[:, 0].copy().reshape(1, -1)))
    dynamicPath_y = np.vstack((dynamicPath_y, dynamicNew[:, 1].copy().reshape(1, -1)))  # 修正后的赋值

    # 更新距离ToGoal和停止条件
    distanceToGoal = np.linalg.norm(robotOld - robotGoal, axis=1)
    stoppingCriteria = max(distanceToGoal)

    plt.pause(0.01)

    # 清空可视化容器
    for circle in dynamic_circles:
        circle.remove()
    dynamic_circles.clear()

    for circle in robot_circles:
        circle.remove()
    robot_circles.clear()

    # 清除之前的机器人连线
    for line in robot_lines:
        line.remove()  # 移除之前的连线
    robot_lines.clear()  # 清空列表
    drawn_lines.clear()

    for line in A_starlines:
        line.remove()
    A_starlines.clear()

# 循环结束
plt.ioff()
plt.show()
