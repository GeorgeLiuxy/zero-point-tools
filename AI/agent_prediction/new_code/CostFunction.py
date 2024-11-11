import numpy as np


def myCost(solution, minimumVelocity, maximumVelocity, minimumTheta, maximumTheta, robotOld, dynamicOld, totalRobot,
           robotRadius, localrobotGoal, grid_map, robotGoal, totalDynamic, dynamicRadius, currentRobot, num):
    # print(static)
    # 初始化机器人新位置的数组
    # robotNew_x = np.zeros(totalRobot)
    # robotNew_y = np.zeros(totalRobot)
    robotNew = currentRobot.copy()
    # 初始化适应度数组
    fit1 = fit2 = fit3 = fit4 = fit5 = 0
    robot = robotOld.copy()

    # 遍历每个机器人
    # for i in range(totalRobot):
    #     # 机器人相对于相应解决方案的新位置
    solutioncopy = solution.copy()
    # print(robotOld[:,0])
    solutioncopy[1] = np.interp(solutioncopy[1], [minimumVelocity, maximumVelocity],
                                [minimumTheta, maximumTheta])

    robotNew[0] = currentRobot[0] + solutioncopy[0] * np.cos(solutioncopy[1])
    robotNew[1] = currentRobot[1] + solutioncopy[0] * np.sin(solutioncopy[1])

    robot[num] = robotNew.copy()

    # F1 (最短距离)
    if np.linalg.norm(robotNew - robotGoal) < 1:
        fit1 -= 100
    else:
        fit1 = 10*(np.sqrt((robotNew[0] - currentRobot[0]) ** 2 + (robotNew[1] - currentRobot[1]) ** 2) + \
              np.sqrt((robotNew[0] - localrobotGoal[0]) ** 2 + (robotNew[1] - localrobotGoal[1]) ** 2))
    # print(fit)

    securityDistance = 0.5
    # F2 (静态障碍物避障)
    cx, cy = robotNew
    radius = robotRadius[0]
    for i in range(int(cy - 0.5), int(cy + 0.5) + 1):
        for j in range(int(cx - 0.5), int(cx + 0.5) + 1):
            if i >= 0 and i < grid_map.shape[0] and j >= 0 and j < grid_map.shape[1]:
                if grid_map[i, j] == 1:
                    fit2 += 50

    # F3 (动态障碍物避障)
    for j in range(totalDynamic):
        distanceToDynamic = np.sqrt((robotNew[0] - dynamicOld[j, 0]) ** 2 + (robotNew[1] - dynamicOld[j, 1]) ** 2)
        if distanceToDynamic < dynamicRadius[j] + radius + securityDistance:
            fit3 += 50

    # F4 (机器人间的避障)
    for j in range(totalRobot):
        if num == j:
            continue
        else:
            distanceToRobot = np.sqrt((robotNew[0] - robotOld[j, 0]) ** 2 + (robotNew[1] - robotOld[j, 1]) ** 2)
            if distanceToRobot < radius * 3:
                fit4 += 50

    # # F5 (通信协同)
    # robot_links = np.zeros((totalRobot, totalRobot))
    # robot_snrs = np.zeros((totalRobot, totalRobot))
    # robot_distances = np.zeros((totalRobot, totalRobot))
    # for i in range(totalRobot):
    #     for j in range(totalRobot):
    #         if i == j:
    #             robot_distances[i, j] = np.inf
    #         else:
    #             robot_distances[i, j] = np.sqrt(
    #                 (robot[i, 0] - robot[j, 0]) ** 2 + (robot[i, 1] - robot[j, 1]) ** 2)
    #         robot_links[i, j], capacity, los, robot_snrs[i, j] = is_link_active(robot_distances[i, j], )
    # percent = transitive_closure(robot_links)
    # fit5 -= percent*0.2
    # print(fit5)
    fitness = fit1 + fit2 + fit3 + fit4
    return fitness