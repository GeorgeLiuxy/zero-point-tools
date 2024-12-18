import docplex.mp.model as cpx
import numpy as np
import math
import matplotlib.pyplot as plt

# 保持原有参数定义不变
I = list(range(10))
J = list(range(15))
K = {i: list(range(6)) for i in I}
b = 6000
L = list(range(0, 4))
f = (100, 200, 300, 400, 500, 450, 470, 390, 390, 300)
v = (0, 30, 50, 80)
r = 20
A_min = (0.1, 0.1, 0.1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1)
lambda_param = (3, 4, 5, 6, 3, 3, 4, 5, 6, 3, 3, 4, 5, 2, 1)
service_rate = (0, 11, 11.5, 12)
gamma_values = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
Wq_max = 1000

I_group1 = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19)]
J_group2 = [(7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (10, 11), (12, 13),
            (14, 15), (16, 17), (18, 19)]

# 计算覆盖关系
I1_indices = {}
J2_indices = {}

for j in range(len(J_group2)):
    I_group1_i = []
    for i, point in enumerate(I_group1):
        if math.sqrt((J_group2[j][0] - point[0]) ** 2 + (J_group2[j][1] - point[1]) ** 2) <= r:
            I_group1_i.append(i)
    I1_indices[j] = I_group1_i

for i in range(len(I_group1)):
    J_group2_j = []
    for j, point in enumerate(J_group2):
        if math.sqrt((point[0] - I_group1[i][0]) ** 2 + (point[1] - I_group1[i][1]) ** 2) <= r:
            J_group2_j.append(j)
    J2_indices[i] = J_group2_j


def solve_with_cplex():
    # 创建模型
    model = cpx.Model(name='facility_location')

    # 基本决策变量
    y = model.binary_var_list(len(I), name='y')
    x = {(i, j): model.binary_var(name=f'x_{i}_{j}')
         for i in I for j in J}
    k = model.integer_var_dict(I, lb=0, ub=5, name='k')
    z = {(i, l): model.binary_var(name=f'z_{i}_{l}')
         for i in I for l in L}

    # 辅助变量
    kz = {(i, l): model.continuous_var(lb=0, ub=5, name=f'kz_{i}_{l}')
          for i in I for l in L}

    mu = model.continuous_var_dict(I, lb=0.1, name='mu')
    lambda_i = model.continuous_var_dict(I, lb=0, name='lambda_i')
    rho = model.continuous_var_dict(I, lb=0, ub=0.99, name='rho')
    W = model.continuous_var_dict(I, lb=0, name='W')

    # 新增辅助变量
    Wmu = model.continuous_var_dict(I, lb=0, name='Wmu')  # W[i] * mu[i]
    rho_lambda = model.continuous_var_dict(I, lb=0, name='rho_lambda')  # rho[i] * lambda_i[i]
    W_lambda = model.continuous_var_dict(I, lb=0, name='W_lambda')  # W[i] * lambda_i[i]

    # McCormick 线性化约束
    for i in I:
        for l in L:
            model.add_constraint(kz[i, l] <= 5 * z[i, l])
            model.add_constraint(kz[i, l] <= k[i])
            model.add_constraint(kz[i, l] >= k[i] - 5 * (1 - z[i, l]))
            model.add_constraint(kz[i, l] >= 0)

    # 服务率约束
    for i in I:
        model.add_constraint(
            mu[i] == model.sum(service_rate[l] * kz[i, l] for l in L)
        )

    # 到达率约束
    for i in I:
        model.add_constraint(
            lambda_i[i] == model.sum(lambda_param[j] * x[i, j] for j in J2_indices[i])
        )

    # 利用率约束
    for i in I:
        model.add_constraint(lambda_i[i] == mu[i] * rho[i])
        model.add_constraint(lambda_i[i] <= mu[i])  # 确保稳定性

    # 预算约束
    model.add_constraint(
        model.sum(f[i] * y[i] + model.sum(v[l] * kz[i, l] for l in L)
                  for i in I) <= b
    )

    # 需求分配约束
    for i in I:
        for j in J2_indices[i]:
            model.add_constraint(x[i, j] <= y[i])

    # 单一分配约束
    for j in J:
        model.add_constraint(
            model.sum(x[i, j] for i in I1_indices[j]) <= 1
        )

    # 技术选择约束
    for i in I:
        model.add_constraint(
            model.sum(z[i, l] for l in L) == y[i]
        )

    # 设备数量约束
    for i in I:
        model.add_constraint(k[i] <= y[i] * max(K[i]))
        model.add_constraint(k[i] >= y[i])  # 至少一个设备

    # 距离约束
    for j in J:
        for i in I1_indices[j]:
            closer_facilities = [o for o in I1_indices[j]
                                 if math.sqrt((J_group2[j][0] - I_group1[o][0]) ** 2 +
                                              (J_group2[j][1] - I_group1[o][1]) ** 2) <=
                                 math.sqrt((J_group2[j][0] - I_group1[i][0]) ** 2 +
                                           (J_group2[j][1] - I_group1[i][1]) ** 2)]
            model.add_constraint(
                model.sum(x[o, j] for o in closer_facilities) + y[i] <= 1
            )

    # 最小需求约束
    for i in I:
        model.add_constraint(lambda_i[i] >= A_min[i] * y[i])

    # 等待时间约束的线性化
    M = 10000  # 一个足够大的数
    for i in I:
        # McCormick envelope for Wmu[i] = W[i] * mu[i]
        model.add_constraint(Wmu[i] <= M * mu[i])
        model.add_constraint(Wmu[i] <= W[i] * M)
        model.add_constraint(Wmu[i] >= W[i] * mu[i] - M * (1 - y[i]))

        # McCormick envelope for rho_lambda[i] = rho[i] * lambda_i[i]
        model.add_constraint(rho_lambda[i] <= M * lambda_i[i])
        model.add_constraint(rho_lambda[i] <= rho[i] * M)
        model.add_constraint(rho_lambda[i] >= rho[i] * lambda_i[i] - M * (1 - y[i]))

        # McCormick envelope for W_lambda[i] = W[i] * lambda_i[i]
        model.add_constraint(W_lambda[i] <= M * lambda_i[i])
        model.add_constraint(W_lambda[i] <= W[i] * M)
        model.add_constraint(W_lambda[i] >= W[i] * lambda_i[i] - M * (1 - y[i]))

        # 等待时间主约束 (分段线性化)
        num_segments = 5
        for s in range(num_segments):
            rho_point = s / num_segments
            if rho_point < 0.99:
                model.add_constraint(
                    Wmu[i] >= lambda_i[i] + rho_lambda[i]
                )

    # 目标函数
    model.minimize(model.sum(W_lambda[i] for i in I))

    # 求解
    solution = model.solve(log_output=True)

    if solution:
        # 提取结果
        y_sol = [solution.get_value(y[i]) for i in I]
        x_sol = [[solution.get_value(x[i, j]) for j in J] for i in I]
        k_sol = {i: solution.get_value(k[i]) for i in I}
        z_sol = [[solution.get_value(z[i, l]) for l in L] for i in I]
        W_sol = {i: solution.get_value(W[i]) for i in I}
        lambda_sol = {i: solution.get_value(lambda_i[i]) for i in I}
        mu_sol = {i: solution.get_value(mu[i]) for i in I}
        rho_sol = {i: solution.get_value(rho[i]) for i in I}

        print("\n最优解：")
        print(f"设施开放状态 (y): {y_sol}")
        print(f"需求分配 (x): {x_sol}")
        print(f"设备数量 (k): {k_sol}")
        print(f"技术选择 (z): {z_sol}")
        print(f"等待时间 (W): {W_sol}")
        print(f"到达率 (lambda): {lambda_sol}")
        print(f"服务率 (mu): {mu_sol}")
        print(f"利用率 (rho): {rho_sol}")
        print(f"目标函数值: {solution.get_objective_value()}")

        return y_sol, x_sol, k_sol, z_sol, solution.get_objective_value()
    else:
        print("模型无解")
        return None


# 运行求解器
result = solve_with_cplex()

# 如果有解，绘制结果图
if result:
    y_sol, x_sol, k_sol, z_sol, obj_value = result

    plt.figure(figsize=(12, 8))

    # 绘制设施点
    for i in I:
        if y_sol[i] > 0.5:
            plt.plot(I_group1[i][0], I_group1[i][1], 'rs', markersize=10, label='开放设施' if i == 0 else "")
        else:
            plt.plot(I_group1[i][0], I_group1[i][1], 'ks', markersize=10, label='关闭设施' if i == 0 else "")

    # 绘制需求点和分配关系
    for j in J:
        plt.plot(J_group2[j][0], J_group2[j][1], 'bo', markersize=8, label='需求点' if j == 0 else "")
        for i in I:
            if x_sol[i][j] > 0.5:
                plt.plot([I_group1[i][0], J_group2[j][0]],
                         [I_group1[i][1], J_group2[j][1]],
                         'g--', alpha=0.5)

    plt.title('设施选址和需求分配结果')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.legend()
    plt.grid(True)
    plt.show()