
import matplotlib.pyplot as plt

# 假设路径规划的结果是以下点
path = [(0, 0), (1, 2), (3, 4), (5, 6)]
obstacles = [(2, 3), (4, 5)]  # 障碍物位置

# 绘制路径
path_x, path_y = zip(*path)
plt.plot(path_x, path_y, label="Planned Path", color='blue')

# 绘制障碍物
obstacle_x, obstacle_y = zip(*obstacles)
plt.scatter(obstacle_x, obstacle_y, label="Obstacles", color='red')

# 显示图例和图形
plt.legend()
plt.show()
