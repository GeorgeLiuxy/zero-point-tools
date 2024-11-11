# 这是一个简化的PID控制器框架，用于更精确地控制机器人
from new_code.main import goal_direction, bestSol


class PIDController:
    def __init__(self, p, i, d):
        self.p = p
        self.i = i
        self.d = d
        self.last_error = 0
        self.integral = 0

    def compute(self, error):
        self.integral += error
        derivative = error - self.last_error
        output = self.p * error + self.i * self.integral + self.d * derivative
        self.last_error = error
        return output

pid = PIDController(p=0.1, i=0.01, d=0.01)
angle_error = goal_direction - bestSol[1]
control_signal = pid.compute(angle_error)
bestSol[1] += control_signal  # 调整角度
