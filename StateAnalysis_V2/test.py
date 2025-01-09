import numpy as np
import plotly.graph_objects as go

# 地球半径，单位为km
earth_radius = 6371

# 卫星数量
num_satellites = 5

# 每颗卫星轨迹的点数
num_points = 200

# 动画帧数
num_frames = 200

# 初始轨道半长轴（单位：km）
initial_semi_major_axis = 7000

# 模拟轨迹数据生成函数
def generate_trajectory(num_points=200, mode="keep", semi_major_axis=7000):
    trajectory = []
    for t in range(num_points):
        if mode == "increase":
            semi_major_axis += np.random.uniform(0.1, 0.3)  # 升轨：轨道半长轴逐渐增大
        elif mode == "decrease":
            semi_major_axis -= np.random.uniform(0.1, 0.3)  # 降轨：轨道半长轴逐渐减小
        else:
            semi_major_axis += np.random.uniform(-0.05, 0.05)  # 受控保持：轨道半长轴波动

        # 假设地球上的轨道路径：纬度、经度
        latitude = np.sin(np.radians(semi_major_axis / 7000.0 * 180)) * 90  # 简化的纬度计算
        longitude = (t * 360 / num_points) % 360  # 经度随着时间变化

        trajectory.append((latitude, longitude, semi_major_axis))

    return trajectory

# 生成多个卫星的轨迹数据
def generate_multiple_trajectories(num_satellites=5, num_points=200):
    trajectories = []
    for i in range(num_satellites):
        mode = np.random.choice(["increase", "decrease", "keep"])  # 随机选择轨道模式
        trajectory = generate_trajectory(num_points=num_points, mode=mode, semi_major_axis=initial_semi_major_axis + i * 100)
        trajectories.append(trajectory)
    return trajectories

# 创建地球模型
def create_earth():
    # 生成一个球形地球模型
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)

    # 球体的坐标（使用球坐标系转换为笛卡尔坐标）
    x = earth_radius * np.sin(theta) * np.cos(phi)
    y = earth_radius * np.sin(theta) * np.sin(phi)
    z = earth_radius * np.cos(theta)

    # 绘制地球
    earth = go.Surface(
        x=x, y=y, z=z,
        colorscale='Blues',
        opacity=0.6,
        showscale=False
    )
    return earth

# 创建卫星轨迹的动画
def create_trajectory_animation(trajectories, num_frames=200):
    fig = go.Figure()

    # 添加地球模型
    fig.add_trace(create_earth())

    # 添加轨迹线
    lines = []
    for i, trajectory in enumerate(trajectories):
        latitudes = [point[0] for point in trajectory]
        longitudes = [point[1] for point in trajectory]
        semi_major_axes = [point[2] for point in trajectory]

        # 将轨迹转换为 3D 经纬度坐标
        x = [earth_radius * np.cos(np.radians(lat)) * np.cos(np.radians(lon)) for lat, lon in zip(latitudes, longitudes)]
        y = [earth_radius * np.cos(np.radians(lat)) * np.sin(np.radians(lon)) for lat, lon in zip(latitudes, longitudes)]
        z = [earth_radius * np.sin(np.radians(lat)) for lat in latitudes]

        # 创建每颗卫星的轨迹线
        line = go.Scatter3d(
            x=x[:1], y=y[:1], z=z[:1],
            mode='markers+lines',
            marker=dict(size=5, color=np.random.rand(), colorscale="Viridis", showscale=True),
            line=dict(width=2, color='blue'),
            name=f"Satellite {i+1}"
        )
        lines.append(line)
        fig.add_trace(line)

    # 动画更新函数
    def update_trace(frame):
        for i, trajectory in enumerate(trajectories):
            latitudes = [point[0] for point in trajectory]
            longitudes = [point[1] for point in trajectory]
            semi_major_axes = [point[2] for point in trajectory]

            x = [earth_radius * np.cos(np.radians(latitudes[frame])) * np.cos(np.radians(longitudes[frame]))]
            y = [earth_radius * np.cos(np.radians(latitudes[frame])) * np.sin(np.radians(longitudes[frame]))]
            z = [earth_radius * np.sin(np.radians(latitudes[frame]))]

            lines[i].update(x=x, y=y, z=z)
        return lines

    # 创建动画帧
    frames = [go.Frame(
        data=[line],
        name=str(i)
    ) for i in range(1, num_frames+1)]

    fig.frames = frames

    # 设置动画播放
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-7000, 7000], title="X"),
            yaxis=dict(range=[-7000, 7000], title="Y"),
            zaxis=dict(range=[-7000, 7000], title="Z"),
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]
            )]
        )]
    )

    # 显示图形
    fig.show()

# 生成轨迹数据
trajectories = generate_multiple_trajectories(num_satellites=num_satellites, num_points=num_points)

# 创建动画
create_trajectory_animation(trajectories, num_frames=num_frames)
