import pygame
import random

# 初始化pygame
pygame.init()

# 游戏设置
WIDTH, HEIGHT = 600, 600  # 窗口大小
GRID_SIZE = 30  # 网格大小
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE
FPS = 10  # 帧率

# 颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# 游戏窗口
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("多智能体路径规划")

# 定义智能体类
class Agent:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.goal = None

    def set_goal(self, goal_x, goal_y):
        self.goal = (goal_x, goal_y)

    def move(self, direction):
        if direction == 0:  # 上
            self.y -= 1
        elif direction == 1:  # 右上
            self.x += 1
            self.y -= 1
        elif direction == 2:  # 右
            self.x += 1
        elif direction == 3:  # 右下
            self.x += 1
            self.y += 1
        elif direction == 4:  # 下
            self.y += 1
        elif direction == 5:  # 左下
            self.x -= 1
            self.y += 1
        elif direction == 6:  # 左
            self.x -= 1
        elif direction == 7:  # 左上
            self.x -= 1
            self.y -= 1

    def draw(self):
        pygame.draw.circle(screen, self.color, (self.x * GRID_SIZE + GRID_SIZE // 2, self.y * GRID_SIZE + GRID_SIZE // 2), GRID_SIZE // 3)

# 定义游戏环境
class GameEnvironment:
    def __init__(self, width, height, num_agents):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.grid = [[0] * self.width for _ in range(self.height)]  # 初始化网格
        self.agents = [Agent(random.randint(0, self.width-1), random.randint(0, self.height-1), BLUE) for _ in range(self.num_agents)]
        self.goals = [(random.randint(0, self.width-1), random.randint(0, self.height-1)) for _ in range(self.num_agents)]
        self.obstacles = [(random.randint(0, self.width-1), random.randint(0, self.height-1)) for _ in range(10)]  # 随机障碍物

        for (x, y) in self.obstacles:
            self.grid[y][x] = 1  # 障碍物

    def reset(self):
        self.grid = [[0] * self.width for _ in range(self.height)]
        for (x, y) in self.obstacles:
            self.grid[y][x] = 1  # 重新放置障碍物
        for agent in self.agents:
            agent.x, agent.y = random.randint(0, self.width-1), random.randint(0, self.height-1)
        for i, agent in enumerate(self.agents):
            agent.set_goal(self.goals[i][0], self.goals[i][1])

    def draw(self):
        # 绘制背景
        screen.fill(WHITE)

        # 绘制障碍物
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i][j] == 1:
                    pygame.draw.rect(screen, BLACK, (j * GRID_SIZE, i * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # 绘制智能体和目标
        for agent in self.agents:
            agent.draw()
        for goal in self.goals:
            pygame.draw.rect(screen, GREEN, (goal[0] * GRID_SIZE, goal[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    def step(self):
        # 每个智能体选择一个动作
        for agent in self.agents:
            direction = random.randint(0, 7)  # 随机选择一个方向
            agent.move(direction)

# 主程序
def main():
    clock = pygame.time.Clock()
    env = GameEnvironment(GRID_WIDTH, GRID_HEIGHT, num_agents=2)

    # 游戏主循环
    running = True
    while running:
        clock.tick(FPS)

        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 游戏逻辑
        env.step()  # 更新智能体的位置
        env.draw()  # 绘制游戏画面

        # 更新屏幕
        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()
