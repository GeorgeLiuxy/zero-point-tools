import numpy as np

from scipy.interpolate import CubicSpline
def dasca_f(maximumFitness, populationSize, dimension, maximumVelocity, minimumVelocity, minimumTheta, maximumTheta,
            robotOld, dynamicOld, totalRobot, robotRadius, localrobotGoal, grid_map, robotGoal, totalDynamic,
            dynamicRadius, costFunction, CurrentRobot, num):
    """
    DASCA算法（自适应鲸鱼优化算法）的实现，用于求解机器人路径规划等问题。

    参数说明：
    - maximumFitness: 最大迭代次数，表示算法最多进行多少次迭代
    - populationSize: 种群规模，表示搜索的个体数目
    - dimension: 问题的维度，表示搜索空间的维度（例如，二维空间问题是2）
    - maximumVelocity: 最大速度，表示个体的速度上限
    - minimumVelocity: 最小速度，表示个体的速度下限
    - minimumTheta: 最小角度，表示角度的最小值（假设与方向相关）
    - maximumTheta: 最大角度，表示角度的最大值
    - robotOld: 当前机器人的状态信息（位置、速度等）
    - dynamicOld: 动态障碍物的状态信息
    - totalRobot: 总机器人数量
    - robotRadius: 机器人的半径
    - localrobotGoal: 每个机器人局部目标的位置信息
    - grid_map: 网格地图，用于表示环境的障碍物或可行走区域
    - robotGoal: 机器人的全局目标位置
    - totalDynamic: 总动态障碍物数量
    - dynamicRadius: 动态障碍物的半径
    - costFunction: 适应度函数，用于评估个体的好坏
    - CurrentRobot: 当前机器人索引
    - num: 当前问题的特定标识（例如任务编号）

    返回值：
    - bestSol: 最优解的个体位置
    - bestCost: 最优解的适应度值（成本）
    """

    # 控制参数
    fitcount = 0  # 适应度计算次数
    myCost = costFunction  # 适应度函数
    SearchAgents_no = populationSize  # 种群规模
    Max_iteration = maximumFitness  # 最大迭代次数
    lb = minimumVelocity  # 速度下限
    ub = maximumVelocity  # 速度上限
    dim = dimension  # 问题维度
    a = 2  # 一个控制参数，通常会随着迭代逐渐减小

    # 随机生成初始化种群（每个个体的维度为dim）
    X = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb

    # 初始化目标位置和适应度
    Destination_position = np.zeros(dim)
    Destination_fitness = float('inf')  # 初始适应度设为无穷大
    Convergence_curve = np.zeros(Max_iteration)  # 收敛曲线
    Objective_values = np.zeros(SearchAgents_no)  # 每个个体的适应度值

    # 计算初始化种群的适应度，并找出最优个体
    for i in range(SearchAgents_no):
        # 计算每个个体的适应度
        Objective_values[i] = myCost(X[i, :], minimumVelocity, maximumVelocity, minimumTheta, maximumTheta,
                                     robotOld, dynamicOld, totalRobot, robotRadius, localrobotGoal, grid_map,
                                     robotGoal, totalDynamic, dynamicRadius, CurrentRobot, num)
        fitcount += 1

        # 如果当前个体更优，更新最优解
        if Objective_values[i] < Destination_fitness:
            Destination_position = X[i, :].copy()
            Destination_fitness = Objective_values[i].copy()

    # 主循环，进行多次迭代
    for t in range(1, Max_iteration + 1):
        r1 = a - t * (a / Max_iteration)  # r1线性递减，从a到0
        r1 = np.round(r1, 4)  # 保留4位小数

        # 更新位置的随机参数r2, r3, r4（控制个体更新行为）
        r2 = 2 * np.pi * np.random.rand(SearchAgents_no, dim)  # 方向角度
        r3 = 2 * np.random.rand(SearchAgents_no, dim)  # 距离因子
        r4 = np.random.rand(SearchAgents_no, dim)  # 随机行为选择

        # 保留4位小数
        r2 = np.round(r2, 4)
        r3 = np.round(r3, 4)
        r4 = np.round(r4, 4)

        # 更新每个解的位置
        for i in range(SearchAgents_no):
            for j in range(dim):
                if r4[i, j] < 0.5:
                    # 如果r4小于0.5，使用sin函数更新位置
                    X[i, j] += r1 * np.sin(r2[i, j]) * abs(r3[i, j] * Destination_position[j] - X[i, j])
                else:
                    # 如果r4大于等于0.5，使用cos函数更新位置
                    X[i, j] += r1 * np.cos(r2[i, j]) * abs(r3[i, j] * Destination_position[j] - X[i, j])

                # 保证位置不越界
                X[i, j] = np.round(X[i, j], 4)

            # 检查是否越界，并将解带回合法范围
            Flag4ub = X[i, :] > ub  # 大于上限
            Flag4lb = X[i, :] < lb  # 小于下限
            X[i, :] = (X[i, :] * (~(Flag4ub + Flag4lb))) + ub * Flag4ub + lb * Flag4lb  # 保持在范围内

            # 计算新的适应度值
            Objective_values[i] = myCost(X[i, :], minimumVelocity, maximumVelocity, minimumTheta, maximumTheta,
                                         robotOld, dynamicOld, totalRobot, robotRadius, localrobotGoal, grid_map,
                                         robotGoal, totalDynamic, dynamicRadius, CurrentRobot, num)
            fitcount += 1

            # 如果当前解更优，更新最优解
            if Objective_values[i] < Destination_fitness:
                Destination_position = X[i, :].copy()
                Destination_fitness = Objective_values[i].copy()

        # 记录当前最优解的适应度（收敛曲线）
        Convergence_curve[t - 1] = Destination_fitness

    # 返回最优解及其适应度值
    bestSol = Destination_position.copy()
    bestCost = Destination_fitness

    return bestSol, bestCost


