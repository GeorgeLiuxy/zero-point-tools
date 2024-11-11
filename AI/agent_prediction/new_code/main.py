import matplotlib

from gridsca import dasca_f
from new_code.DQN import DQNAgent

matplotlib.use('TkAgg')  # 确保使用支持交互的后端
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from Correspondence import is_link_active
from NLoS_probability import count_obstacles
from global_path_generator import find_globalpath
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
def compute_adjacency_matrix(robot_positions, communication_range):
    """
    计算机器人之间的通信邻接矩阵。

    :param robot_positions: 一个形状为 (totalRobot, 2) 的数组，表示每个机器人的位置。
    :param communication_range: 机器人之间的通信范围。
    :return: 邻接矩阵，一个形状为 (totalRobot, totalRobot) 的矩阵，
             其中A[i][j]表示机器人i和机器人j是否在通信范围内。
    """
    num_robots = len(robot_positions)
    adjacency_matrix = np.zeros((num_robots, num_robots))  # 初始化邻接矩阵

    # 计算每对机器人之间的距离
    for i in range(num_robots):
        for j in range(num_robots):
            if i != j:
                distance = np.linalg.norm(robot_positions[i] - robot_positions[j])  # 计算欧氏距离
                if distance <= communication_range:  # 如果在通信范围内
                    adjacency_matrix[i][j] = 1  # 机器人i与机器人j可以通信
    return adjacency_matrix


def calculate_dynamic_goal(robot_position, grid_map, robot_goal, avoidance_radius=1.0):
    """
    计算动态目标位置，避免障碍物或其他机器人。

    :param robot_position: 当前机器人位置 (x, y)
    :param grid_map: 环境栅格地图，其中1表示障碍物，0表示空白区域
    :param robot_goal: 机器人的目标位置 (x, y)
    :param avoidance_radius: 避免障碍物的半径

    :return: 动态目标位置 (x, y)
    """
    # 假设机器人当前位置为 robot_position，目标为 robot_goal，避免半径为 avoidance_radius
    x, y = robot_position
    goal_x, goal_y = robot_goal

    # 检查目标位置是否受到障碍物的影响
    # 计算目标与障碍物之间的距离
    distance_to_goal = np.linalg.norm(np.array(robot_position) - np.array(robot_goal))

    # 如果目标位置处于障碍物附近，动态调整目标位置
    if grid_map[int(goal_y)][int(goal_x)] == 1 or distance_to_goal < avoidance_radius:
        print(f"目标位置 {robot_goal} 受阻，正在调整目标...")

        # 如果目标被障碍物挡住，尝试找到一个新的位置（例如远离障碍物）
        # 这里只是一个简单的调整方法，实际情况可能需要更复杂的规划
        direction_to_goal = np.array([goal_x - x, goal_y - y])
        direction_to_goal /= np.linalg.norm(direction_to_goal)  # 归一化方向

        # 向目标方向调整，同时避免障碍物
        new_goal = robot_goal + direction_to_goal * avoidance_radius
        new_goal_x, new_goal_y = new_goal

        # 确保新目标不超出地图边界
        new_goal_x = np.clip(new_goal_x, 0, grid_map.shape[1] - 1)
        new_goal_y = np.clip(new_goal_y, 0, grid_map.shape[0] - 1)

        # 返回调整后的目标位置
        return new_goal_x, new_goal_y
    else:
        # 如果目标位置没有问题，返回原目标位置
        return goal_x, goal_y


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
        localgridmap[y_min:y_max, x_min:x_max] = combined_grid[y_min:y_max, x_min:x_max]

    # 合并更新机器人观测矩阵
    for i in range(totalRobot):
        combined_grid_robot = np.logical_or(grid_robotmaps[i], localgridmap).astype(int)
        grid_robotmaps[i] = combined_grid_robot

    # 初始化强化学习代理
    state_size = 5  # 机器人状态空间大小（例如位置、速度、目标位置等）
    action_size = 5  # 动作空间大小（例如：速度调整、方向变化等）
    agents = [DQNAgent(state_size, action_size) for _ in range(totalRobot)]  # 为每个机器人创建一个 DQN 代理

    for i in range(totalRobot):
        # 使用 A* 算法计算全局路径
        A_starpath = find_globalpath(robotOld[i], robotGoal[i], grid_robotmaps[i])

        # 计算到目标的距离
        distanceToGoal[i] = np.linalg.norm(robotOld[i] - robotGoal[i])

        if len(A_starpath) >= 2:
            # 计算基于距离的比例值，使用指数函数
            proportion = math.exp(-distanceToGoal[i] / initial_distances[i] * 4)
            # 使用比例值计算适应因子
            adaptfactor = 5 + (100 - 5) * proportion
            A_starpath = b_spline_interpolation(A_starpath)

            differences = np.linalg.norm(A_starpath - A_starpath[0], axis=1)
            index = np.argmax(differences > 3) if np.any(differences > 3) else -1
            localrobotGoal = A_starpath[index] if index != -1 else robotGoal[i]
        else:
            A_starpath = [robotOld[i].tolist()]
            localrobotGoal = robotGoal[i]

        A_starpaths.append(A_starpath)

        # 初始化 bestSol
        # bestSol = [0, 0]  # [速度, 角度]，初始化为默认值
        # 假设初始速度是 0.5，初始朝向是目标的方向
        goal_direction = math.atan2(robotGoal[i][1] - robotOld[i][1], robotGoal[i][0] - robotOld[i][0])
        # 检查 A_starpath 是否至少包含两个点
        if len(A_starpath) > 1:
            # 获取路径中的当前点和下一个点
            current_point = robotOld[i]
            next_point = A_starpath[1]  # 选择下一个点
        else:
            # 如果路径只有一个点，使用当前位置或目标位置
            next_point = A_starpath[0]  # 只有一个点时，选择路径中的第一个点作为目标
            print(f"路径只有一个点，使用当前位置或目标位置: {next_point}")

        # 计算两点之间的距离
        distance_to_next_point = np.linalg.norm(next_point - current_point)

        # 根据路径点之间的距离来动态调整速度
        if distance_to_next_point > 5:
            initial_speed = 0.7  # 较远时加速
        elif distance_to_next_point < 2:
            initial_speed = 0.3  # 距离较近时减速
        else:
            initial_speed = 0.5  # 一般情况保持速度

        bestSol = [initial_speed, goal_direction]  # [速度, 角度]，初始化为合适的值
        # bestSol, bestCost = dasca_f(
        #     maximumFitness, populationSize, dimension, maximumVelocity, minimumVelocity,
        #     minimumTheta, maximumTheta, robotNew, dynamicNew, totalRobot, robotRadius,
        #     localrobotGoal, grid_map, robotGoal[i],
        #     totalDynamic, dynamicRadius,
        #     costFunction, robotOld[i], i
        # )
        # 获取机器人的当前状态（位置、速度等）
        state = np.array([robotOld[i][0], robotOld[i][1], bestSol[0], bestSol[1], distanceToGoal[i]])

        # 使用 DQN 代理来选择动作
        action = agents[i].act(state)  # 动作：速度调整、方向变化等

        # 计算目标点方向
        goal_direction = math.atan2(localrobotGoal[1] - robotOld[i][1], localrobotGoal[0] - robotOld[i][0])
        angle_diff = goal_direction - bestSol[1]

        # 动作空间：加速、减速、转向、停止
        if action == 0:
            bestSol[0] += 0.5  # 增加速度
        elif action == 1:
            bestSol[0] -= 0.5  # 减少速度
        elif action == 2:
            bestSol[1] += 0.1  # 向左转（增加偏航角）
        elif action == 3:
            bestSol[1] -= 0.1  # 向右转（减少偏航角）
        else:
            bestSol[0] = 0  # 停止

        # 修正角度，确保机器人始终朝目标前进
        bestSol[1] = goal_direction  # 使机器人朝向目标方向

        # 如果距离目标很近，减速并停止
        if distanceToGoal[i] < 1.0:
            bestSol[0] = 0  # 停止移动
            print(f"机器人 {i} 已接近目标，停止移动。")
            done = True  # 标记机器人已到达目标
        else:
            done = False  # 机器人还没有到达目标
        # 更新机器人的位置，只有当距离目标足够远时才更新
        print(f"机器人 {i} 离终点距离: {distanceToGoal[i]}")
        if distanceToGoal[i] > 1 and not done:  # 如果机器人还没有完全到达目标
            bestSol[1] = np.interp(bestSol[1], [minimumTheta, maximumTheta], [minimumTheta, maximumTheta])
            robotNew[i, 0] = robotOld[i, 0] + bestSol[0] * math.cos(bestSol[1])
            robotNew[i, 1] = robotOld[i, 1] + bestSol[0] * math.sin(bestSol[1])
            print(f"机器人 {i} 新位置: {robotNew[i]}")
        else:
            print(f"机器人 {i} 已到达目标位置，保持当前位置: {robotOld[i]}")

        # 更新 A_starpath 起点为机器人的新位置
        if len(A_starpaths[i]) > 0:
            A_starpaths[i][0][0] = robotNew[i, 0]
            A_starpaths[i][0][1] = robotNew[i, 1]

        # 设置障碍物距离的阈值
        collision_threshold = 0.5  # 假设障碍物碰撞的阈值是 0.5
        # 计算机器人与障碍物的距离并进行惩罚
        collision_penalty = 0
        # 假设障碍物列表是 dynamic_obstacles 或 static_obstacles
        for obstacle in dynamic_circles:  # 遍历动态障碍物
            # 从 Circle 对象中提取坐标和半径
            obstacle.xy = obstacle.get_center()
            obstacle_position = np.array(obstacle.xy)
            obstacle_radius = obstacle.radius

            if obstacle_position.shape != (2,):
                raise ValueError(f"Obstacle should be a 2D coordinate, but got {obstacle_position.shape}")

            # 计算机器人当前位置与障碍物之间的距离
            obstacle_distance = np.linalg.norm(robotNew[i] - obstacle_position)

            # 如果距离小于障碍物的半径，加上碰撞惩罚
            if obstacle_distance < (obstacle_radius + collision_threshold):
                collision_penalty = -20  # 发生碰撞时给予惩罚
                break

        # 计算机器人到目标的距离
        distance_to_goal = np.linalg.norm(robotNew[i] - robotGoal[i])

        # 检查是否越过了目标（即机器人的当前位置比目标位置更远）
        goal_direction = np.arctan2(robotGoal[i][1] - robotOld[i][1], robotGoal[i][0] - robotOld[i][0])
        goal_distance = np.linalg.norm(robotOld[i] - robotGoal[i])
        # 如果机器人越过目标，进行惩罚
        goal_passed_penalty = 0
        if goal_distance > distance_to_goal and np.abs(goal_direction - bestSol[1]) < 0.1:
            goal_passed_penalty = -30  # 对越过终点的行为进行惩罚

        # 计算奖励：离目标越近，奖励越高，碰撞时加上惩罚
        reward = -distance_to_goal  # 奖励：离目标越近，奖励越高
        # 如果有碰撞惩罚，则加上碰撞惩罚
        reward += collision_penalty
        # 如果越过目标，加上惩罚
        reward += goal_passed_penalty
        # 如果距离目标足够近，停止移动并标记为 done
        done = True if distance_to_goal <= 1 else False
        # 输出当前的状态和奖励
        print(f"机器人 {i} 到目标的距离: {distance_to_goal}, 奖励: {reward}")

        # 存储记忆并进行回放
        next_state = np.array([robotNew[i][0], robotNew[i][1], bestSol[0], bestSol[1], distance_to_goal])
        agents[i].remember(state, action, reward, next_state, done)
        agents[i].replay()  # 进行经验回放
        agents[i].update_target_model()  # 更新目标 Q 网络

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
