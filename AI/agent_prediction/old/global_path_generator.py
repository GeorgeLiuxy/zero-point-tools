import heapq
import math

import numpy as np


# 节点类
class Node:
    def __init__(self, index, x, y):
        self.index = index
        self.x = x
        self.y = y
        self.t = 'new'
        self.k = float('inf')
        self.h = float('inf')
        self.parent = None

# 启发式函数，使用欧几里得距离
def heuristic(node, goal_node):
    # return abs(node.x - goal_node.x) + abs(node.y - goal_node.y)
    return math.sqrt((node.x - goal_node.x)**2+(node.y - goal_node.y)**2)

# 堆操作，用于获取最小k值的节点
def push(open_list, node_index, k):
    heapq.heappush(open_list, (k, node_index))


def pop(open_list):
    return heapq.heappop(open_list)[1]


# 主函数
def find_globalpath(start, goal, grid_map):
    grid_size = len(grid_map)  # 假设grid_map是方阵

    # 创建节点列表
    nodes = [Node(i, x, y) for y in range(grid_size) for x in range(grid_size) for i in [y * grid_size + x]]

    # 定义起点和终点节点
    start_node = nodes[int(start[1]) * grid_size + int(start[0])]
    goal_node = nodes[int(goal[1]) * grid_size + int(goal[0])]

    # 初始化open_list和goal_node
    open_list = []
    goal_node.t = 'open'
    goal_node.k = 0
    goal_node.h = 0
    push(open_list, goal_node.index, goal_node.k)

    # 成本函数
    def myCost(x, y, safety_distance=2):
        if grid_map[y, x] == 1:
            return float('inf')  # 障碍物成本为无穷大

        # 检查与障碍物的距离
        for dx in range(-safety_distance + 1, safety_distance):
            for dy in range(-safety_distance + 1, safety_distance):
                if 0 <= x + dx < grid_map.shape[1] and 0 <= y + dy < grid_map.shape[0]:
                    if grid_map[y + dy, x + dx] == 1:
                        # 如果在安全距离内发现障碍物，则增加成本
                        if dx == 0 and dy == 0:
                            # 实际上是障碍物本身，应该已经在上面的检查中被捕获
                            return float('inf')
                        else:
                            # 与障碍物相邻的栅格成本较高
                            return 10  # 可以根据需要调整这个值

        return 1  # 普通栅格成本为1

    # process_state函数的简化版本
    def process_state(open_list, nodes, goal_node_index):
        current_index = pop(open_list)
        current_node = nodes[current_index]
        current_node.t = 'closed'

        # 假设邻居节点为上下左右四个方向
        neighbors = [
            (current_node.x - 1, current_node.y),
            (current_node.x + 1, current_node.y),
            (current_node.x, current_node.y - 1),
            (current_node.x, current_node.y + 1)
        ]

        k_old = current_node.k

        for nx, ny in neighbors:
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                neighbor_index = ny * grid_size + nx
                neighbor_node = nodes[neighbor_index]

                if neighbor_node.t == 'new' or neighbor_node.t == 'open':
                    new_k = min(k_old + myCost(nx, ny), neighbor_node.k)
                    if new_k < neighbor_node.k:
                        neighbor_node.k = new_k
                        neighbor_node.parent = current_node
                        if neighbor_node.t == 'new':
                            neighbor_node.t = 'open'
                            push(open_list, neighbor_index, neighbor_node.k + heuristic(neighbor_node, goal_node))

        return nodes, open_list, k_old

    # 主循环，简化版本
    while True:
        if not open_list:
            return []# 如果open_list为空，说明没有路径可找
        nodes, open_list, k_old = process_state(open_list, nodes, goal_node.index)
        if start_node.t == 'closed':
            break

    # 检查是否找到有效路径
    if start_node.t != 'closed':
        return []  # 如果没有到达起点，返回空列表

    # 重建路径
    path = []
    node = start_node
    while node is not None:
        path.append([node.x, node.y])
        node = node.parent
    # print(path)
    smooth_path = generate_smooth_path(path,grid_map)
    smooth_path_list_updated = [[x + 0.5 for x in sublist] for sublist in smooth_path]
    return smooth_path_list_updated

def generate_smooth_path(path, G):
    path1 = np.copy(path)
    long = path1.shape[0]
    i = 0

    while i < long - 2:
        b1, a1 = int(path1[i, 0]), int(path1[i, 1])
        b3, a3 = int(path1[i + 2, 0]), int(path1[i + 2, 1])

        if a1 < a3:
            if np.all(G[a1:a3 + 1, b1:b3 + 1] == 0) and np.all(G[a1:a3 + 1, b3] == 0) and np.all(
                    G[a1:a3 + 1, int((b1 + b3) / 2)] == 0):
                path1 = np.delete(path1, i + 1, axis=0)
                i -= 1
        else:
            if np.all(G[a3:a1 + 1, b1:b3 + 1] == 0) and np.all(G[a3:a1 + 1, b3] == 0) and np.all(
                    G[a3:a1 + 1, int((b1 + b3) / 2)] == 0):
                path1 = np.delete(path1, i + 1, axis=0)
                i -= 1

        i += 1
        long = path1.shape[0]

    return path1