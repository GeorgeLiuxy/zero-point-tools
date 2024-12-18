from flask import Flask, render_template, request, jsonify
import json
import os
from control import control_window  # 引入控制函数
import pygetwindow as gw

app = Flask(__name__)

# 加载窗口编号映射
def load_window_mapping():
    if os.path.exists('app/window_mapping.json'):
        with open('app/window_mapping.json', 'r') as f:
            return json.load(f)
    return {}

# 保存窗口编号映射
def save_window_mapping(mapping):
    folder = 'app'  # 目标文件夹
    if not os.path.exists(folder):
        os.makedirs(folder)  # 如果文件夹不存在，创建它

    # 保存 JSON 数据
    with open(os.path.join(folder, 'window_mapping.json'), 'w') as f:
        json.dump(mapping, f)

# 首页路由，显示控制界面
@app.route('/')
def index():
    # 加载当前的窗口映射
    window_mapping = load_window_mapping()
    return render_template('index.html', window_mapping=window_mapping)

# 处理控制请求
@app.route('/control', methods=['POST'])
def control():
    window_id = int(request.json.get('window_id'))
    action = request.json.get('action')
    x = request.json.get('x')
    y = request.json.get('y')
    delta = request.json.get('delta')

    # 查找窗口标题
    window_mapping = load_window_mapping()
    window_title = next((title for title, id in window_mapping.items() if id == window_id), None)

    if not window_title:
        return jsonify({"error": "Window not found"})

    result = control_window(window_title, action, x, y, delta)
    return jsonify(result)

# 生成窗口映射的编号（检测投屏窗口）
@app.route('/detect', methods=['POST'])
def detect_window():
    window_mapping = load_window_mapping()

    # 获取所有窗口的标题
    windows = gw.getAllTitles()
    new_window_mapping = {}

    # 为每个窗口分配一个编号
    next_id = max(window_mapping.values(), default=0) + 1
    for window in windows:
        if "投屏" in window:  # 根据窗口标题包含“投屏”进行识别
            new_window_mapping[window] = next_id
            next_id += 1

    # 合并新的窗口映射
    window_mapping.update(new_window_mapping)
    save_window_mapping(window_mapping)

    return jsonify(window_mapping)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)
