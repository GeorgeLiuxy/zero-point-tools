def count_obstacles(start, end, grid_map, grid_size):
    # 计算连线的起点和终点的整数坐标
    start_x, start_y = int(round(start[0])), int(round(start[1]))
    end_x, end_y = int(round(end[0])), int(round(end[1]))

    # 确保坐标在有效范围内
    start_x = max(0, min(start_x, grid_size - 1))
    start_y = max(0, min(start_y, grid_size - 1))
    end_x = max(0, min(end_x, grid_size - 1))
    end_y = max(0, min(end_y, grid_size - 1))

    # 计算连线的斜率（考虑垂直线的情况）
    if start_x != end_x:
        slope = (end_y - start_y) / (end_x - start_x)
    else:
        slope = None

    # 遍历连线上的点，并统计经过的障碍物栅格数量
    obstacle_count = 0

    if slope is not None:
        # 遍历 x 坐标，并计算对应的 y 坐标
        x_min = max(0, min(start_x, end_x))
        x_max = min(grid_size - 1, max(start_x, end_x))
        for x in range(x_min, x_max + 1):
            y = int(round(start_y + slope * (x - start_x)))
            # 确保 x 和 y 坐标在地图范围内
            if 0 <= x < grid_size and 0 <= y < grid_size:
                if grid_map[y, x] == 1:
                    obstacle_count += 1
    else:
        # 垂直线的情况，遍历 y 坐标
        y_min = max(0, min(start_y, end_y))
        y_max = min(grid_size - 1, max(start_y, end_y))
        x = start_x  # 垂直线，x 坐标不变
        for y in range(y_min, y_max + 1):
            # 确保 x 和 y 坐标在地图范围内
            if 0 <= x < grid_size and 0 <= y < grid_size:
                if grid_map[y, x] == 1:
                    obstacle_count += 1

    return obstacle_count
