import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合批量生成数据
import math
import numpy as np
import collections
import pickle
import os
import random

# 导入外部函数和模块
from Correspondence import is_link_active
from NLoS_probability import count_obstacles
from global_path_generator import find_globalpath
from gridsca import dasca_f
from spline_utilities import b_spline_interpolation
from CostFunction import myCost

# 设置生成数据的参数
NUM_SIMULATIONS = 1000  # 仿真次数，可以调整以生成更多数据
MAX_TIME_STEPS = 50     # 每次仿真的最大时间步数
SAVE_INTERVAL = 100     # 每隔多少次仿真保存一次数据
DATA_DIR = 'simulation_data'  # 数据保存目录

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 定义数据存储结构
all_data = []

# 定义获取局部地图的函数
def get_local_map(position, combined_grid, map_size=11):
    x, y = position
    x, y = int(round(x)), int(round(y))
    half_size = map_size // 2
    x_min = max(0, x - half_size)
    x_max = min(combined_grid.shape[1], x + half_size + 1)
    y_min = max(0, y - half_size)
    y_max = min(combined_grid.shape[0], y + half_size + 1)
    local_map = combined_grid[y_min:y_max, x_min:x_max]
    # 如果边界不足，进行填充
    pad_x_before = max(0, half_size - x)
    pad_x_after = max(0, (x + half_size + 1) - combined_grid.shape[1])
    pad_y_before = max(0, half_size - y)
    pad_y_after = max(0, (y + half_size + 1) - combined_grid.shape[0])
    local_map = np.pad(local_map, ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after)), 'constant', constant_values=0)
    return local_map

for simulation in range(NUM_SIMULATIONS):
    print(f"开始仿真 {simulation + 1}/{NUM_SIMULATIONS}")

    # 随机化机器人数量、起始位置、目标位置、障碍物等

    # 随机生成机器人数量（例如 5 到 10 个机器人）
    totalRobot = random.randint(5, 10)
    robotRadius = np.ones(totalRobot) / 2  # 机器人半径

    # 随机生成机器人的起始位置和目标位置，确保它们不在障碍物内
    def generate_positions(num, grid_size, obstacles):
        positions = []
        while len(positions) < num:
            pos = np.random.uniform(low=0, high=grid_size, size=2)
            in_obstacle = False
            for center, radius in obstacles:
                if np.linalg.norm(pos - np.array(center)) <= radius:
                    in_obstacle = True
                    break
            if not in_obstacle:
                positions.append(pos)
        return np.array(positions)

    # 随机生成障碍物
    num_obstacles = random.randint(5, 15)
    grid_size = 100
    grid_map = np.zeros((grid_size, grid_size))
    obstacles = []
    for _ in range(num_obstacles):
        cx = random.uniform(0, grid_size)
        cy = random.uniform(0, grid_size)
        radius = random.uniform(2, 6)
        obstacles.append(((cx, cy), radius))

    # 绘制障碍物到栅格地图
    for center, radius in obstacles:
        cx, cy = center
        x_min = max(0, int(cx - radius))
        x_max = min(grid_size, int(cx + radius))
        y_min = max(0, int(cy - radius))
        y_max = min(grid_size, int(cy + radius))
        Y, X = np.ogrid[:grid_size, :grid_size]
        dist_from_center = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = dist_from_center <= radius
        grid_map[mask] = 1.0

    # 生成机器人起点和终点
    robotStart = generate_positions(totalRobot, grid_size, obstacles)
    robotGoal = generate_positions(totalRobot, grid_size, obstacles)

    # 为每个机器人随机分配颜色（用于可视化，可选）
    colors = ['#%06X' % random.randint(0, 0xFFFFFF) for _ in range(totalRobot)]

    # 随机生成动态障碍物的数量、起点和终点
    totalDynamic = random.randint(3, 7)
    dynamicStart = generate_positions(totalDynamic, grid_size, obstacles)
    dynamicGoal = generate_positions(totalDynamic, grid_size, obstacles)
    dynamicRadius = np.random.uniform(low=0.5, high=2.0, size=totalDynamic)
    dynamicVelocity = np.random.uniform(low=0.2, high=1.0, size=totalDynamic)

    # 初始化机器人和动态障碍物的位置
    robotOld = robotStart.copy()
    dynamicOld = dynamicStart.copy()

    # 初始化控制参数
    maximumFitness = 200
    populationSize = 50
    dimension = 2
    maximumVelocity = 1
    minimumVelocity = 0.5
    minimumTheta = 0
    maximumTheta = 2 * math.pi
    step = 0

    # 定义初始距离和停止条件
    distanceToGoal = np.linalg.norm(robotOld - robotGoal, axis=1)
    stoppingCriteria = max(distanceToGoal)
    robotNew = robotOld.copy()
    dynamicNew = np.zeros_like(dynamicOld)

    # 初始化机器人路径记录为二维数组
    robotPath_x = robotOld[:, 0].reshape(1, -1)
    robotPath_y = robotOld[:, 1].reshape(1, -1)

    # 初始化动态障碍物路径记录为二维数组
    dynamicPath_x = dynamicOld[:, 0].reshape(1, -1)
    dynamicPath_y = dynamicOld[:, 1].reshape(1, -1)

    # 初始化机器人的历史轨迹队列，队列长度可以根据需要调整
    history_length = 10
    robot_historys = [collections.deque(maxlen=history_length) for _ in range(totalRobot)]

    # 定义初始距离
    initial_distances = np.linalg.norm(robotOld - robotGoal, axis=1)

    # 定义局部地图
    localgridmap = np.zeros_like(grid_map)

    # 初始化数据存储
    data = {
        'timesteps': [],
        'agent_states': [],  # 每个时间步的智能体状态列表
        'adjacency_matrices': [],  # 每个时间步的邻接矩阵
        'edge_features': [],  # 每个时间步的边特征
        # 可选的环境信息
        'dynamic_obstacles': [],  # 动态障碍物状态
        'static_obstacles': grid_map.copy(),  # 静态障碍物地图
        'totalRobot': totalRobot,
        'robotRadius': robotRadius,
        'robotGoal': robotGoal.copy(),  # 添加目标位置
        'robotStart': robotStart.copy(),  # 添加起始位置
    }

    # 主循环
    while stoppingCriteria > 1 and step < MAX_TIME_STEPS:
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
            else:
                dynamicNew[i] = dynamicOld[i]  # 停止移动

            # 更新动态障碍物路径记录
            dynamicPath_x = np.vstack((dynamicPath_x, dynamicNew[:, 0].copy().reshape(1, -1)))
            dynamicPath_y = np.vstack((dynamicPath_y, dynamicNew[:, 1].copy().reshape(1, -1)))

        # 更新栅格图以反映动态障碍物所占的栅格
        grid_dynamicmap = np.zeros((grid_size, grid_size))
        for index, pos in enumerate(dynamicNew):
            x, y = pos
            radius = dynamicRadius[index]
            Y, X = np.ogrid[:grid_size, :grid_size]
            dist_from_center = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
            mask = dist_from_center <= radius
            grid_dynamicmap[mask] = 1

        # 更新全局图
        combined_grid = np.logical_or(grid_map, grid_dynamicmap).astype(int)

        # 更新栅格图以反映机器人所占的栅格
        for i in range(totalRobot):
            grid_robotmap = np.zeros((grid_size, grid_size))
            for j in range(totalRobot):
                if i != j:
                    x, y = robotOld[j]
                    radius = robotRadius[j]
                    Y, X = np.ogrid[:grid_size, :grid_size]
                    dist_from_center = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
                    mask = dist_from_center <= radius
                    grid_robotmap[mask] = 1
            grid_robotmaps.append(grid_robotmap)

        # 提取当前每个机器人观测图
        local_maps = []
        for i in range(totalRobot):
            local_map = get_local_map(robotNew[i], combined_grid)
            local_maps.append(local_map)

        # 初始化存储当前时间步的数据
        timestep_data = {
            'positions': robotNew.copy(),
            'velocities': [],
            'directions': [],
            'accelerations': [],
            'agent_attributes': [],  # 可根据需要添加
            'history_sequences': [],  # 每个智能体的历史轨迹序列
            'local_maps': [],         # 每个智能体的局部地图
            'goal_positions': robotGoal.copy(),  # 终点坐标
            'actions': [],  # 当前时间步的动作（速度、转向角）
        }

        # 计算机器人之间的通信链路和边特征
        num_robots = len(robotNew)
        robot_distances = np.zeros((num_robots, num_robots))
        robot_links = np.zeros((num_robots, num_robots))
        robot_snrs = np.zeros((num_robots, num_robots))
        edge_feature_matrix = []
        communication_masks = np.ones((num_robots, num_robots))  # 通信掩码，1表示通信正常，0表示通信受限
        for i in range(num_robots):
            edge_features_i = []
            for j in range(num_robots):
                if i == j:
                    robot_distances[i, j] = np.inf
                    p_los = 1
                else:
                    robot_distances[i, j] = np.linalg.norm(robotNew[i] - robotNew[j])
                    p_los = count_obstacles(robotNew[i], robotNew[j], grid_map, grid_size) / 10
                link_active, capacity, snr, p_los = is_link_active(robot_distances[i, j], p_los)
                robot_links[i, j] = link_active
                robot_snrs[i, j] = snr

                # 计算相对角度
                if i != j:
                    relative_angle = math.atan2(
                        robotNew[j, 1] - robotNew[i, 1],
                        robotNew[j, 0] - robotNew[i, 0]
                    )
                else:
                    relative_angle = 0

                edge_feature = {
                    'distance': robot_distances[i, j],
                    'relative_angle': relative_angle,
                    'snr': snr,
                    'p_los': p_los,
                    'link_active': link_active,
                    'capacity': capacity,
                }
                edge_features_i.append(edge_feature)

                # 模拟通信受限导致的信息丢失
                packet_loss_rate = 0.1  # 10%的数据包丢失率
                if link_active and random.random() < packet_loss_rate:
                    communication_masks[i, j] = 0  # 通信受限

            edge_feature_matrix.append(edge_features_i)

        # 机器人路径规划和位置更新
        velocities = []
        directions = []
        accelerations = []
        actions = []
        for i in range(totalRobot):
            # 使用A-star算法计算全局路径
            A_starpath = find_globalpath(robotOld[i], robotGoal[i], grid_robotmaps[i])

            # 计算到目标的距离
            distanceToGoal_i = np.linalg.norm(robotOld[i] - robotGoal[i])

            if len(A_starpath) >= 2:
                # 计算一个基于距离的比例值，使用指数函数
                proportion = math.exp(-distanceToGoal_i / initial_distances[i] * 4)
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
                myCost, robotOld[i], i
            )
            # print(f"机器人 {i} 的最佳解决方案: 速度={bestSol[0]}, 角度={bestSol[1]}, 适应度={bestCost}")

            # 如果距离大于1，则更新机器人的位置
            if distanceToGoal_i > 1:
                # 插值运算，更新机器人的转向角信息
                bestSol[1] = np.interp(bestSol[1], [minimumVelocity, maximumVelocity], [minimumTheta, maximumTheta])
                new_x = robotOld[i, 0] + bestSol[0] * math.cos(bestSol[1])
                new_y = robotOld[i, 1] + bestSol[0] * math.sin(bestSol[1])
                robotNew[i, 0] = new_x
                robotNew[i, 1] = new_y

            # 记录速度、方向、加速度
            velocity = bestSol[0]
            direction = bestSol[1]
            if step > 1:
                prev_velocity = data['agent_states'][-1]['velocities'][i]
                acceleration = (velocity - prev_velocity)  # 假设时间步长为1
            else:
                acceleration = 0  # 第一个时间步的加速度设为0
            velocities.append(velocity)
            directions.append(direction)
            accelerations.append(acceleration)
            actions.append([velocity, direction])

            # 更新机器人的历史动作轨迹队列
            robot_historys[i].append({
                'position': robotNew[i].copy(),
                'velocity': velocity,
                'direction': direction,
            })

        # 更新机器人旧位置
        robotOld = robotNew.copy()

        # 更新机器人路径记录
        robotPath_x = np.vstack((robotPath_x, robotNew[:, 0].copy().reshape(1, -1)))
        robotPath_y = np.vstack((robotPath_y, robotNew[:, 1].copy().reshape(1, -1)))

        # 模拟通信受限导致的历史轨迹序列和局部地图信息丢失
        for i in range(totalRobot):
            history_sequence = list(robot_historys[i])  # 转换为列表
            # 遍历所有其他智能体，模拟从其他智能体接收信息
            received_histories = []
            received_local_maps = []
            for j in range(totalRobot):
                if i != j:
                    # 判断是否能接收到来自j的历史序列
                    if robot_links[j, i] and communication_masks[j, i]:
                        # 接收到完整的历史序列
                        received_histories.append(list(robot_historys[j]))
                        received_local_maps.append(local_maps[j])
                    else:
                        # 信息丢失，接收到None
                        received_histories.append(None)
                        received_local_maps.append(None)
                else:
                    # 自己的历史序列和局部地图
                    received_histories.append(history_sequence)
                    received_local_maps.append(local_maps[i])

            # 将接收到的历史序列和局部地图存储
            timestep_data['history_sequences'].append(received_histories)
            timestep_data['local_maps'].append(received_local_maps)

        # 存储其他数据
        timestep_data['velocities'] = velocities
        timestep_data['directions'] = directions
        timestep_data['accelerations'] = accelerations
        timestep_data['actions'] = actions
        # 如果有其他智能体属性，可以添加到 timestep_data['agent_attributes']

        # 记录当前时间步的数据
        data['timesteps'].append(step)
        data['agent_states'].append(timestep_data)
        data['adjacency_matrices'].append(robot_links.copy())
        data['edge_features'].append(edge_feature_matrix)
        data['dynamic_obstacles'].append(dynamicNew.copy())

        # 更新动态障碍物位置
        dynamicOld = dynamicNew.copy()

        # 更新距离ToGoal和停止条件
        distanceToGoal = np.linalg.norm(robotOld - robotGoal, axis=1)
        stoppingCriteria = max(distanceToGoal)

    # 将当前仿真的数据添加到总数据列表
    all_data.append(data)

    # 每隔 SAVE_INTERVAL 次仿真，保存一次数据
    if (simulation + 1) % SAVE_INTERVAL == 0 or (simulation + 1) == NUM_SIMULATIONS:
        # 保存数据到文件
        file_path = os.path.join(DATA_DIR, f'simulation_data_{simulation + 1}.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(all_data, f)
        print(f"数据已保存到 {file_path}")
        # 清空 all_data，释放内存
        all_data = []

print("数据生成完成。")
