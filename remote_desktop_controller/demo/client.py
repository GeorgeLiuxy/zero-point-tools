from pynput.mouse import Listener as MouseListener
from pynput.keyboard import Listener as KeyboardListener, Controller as KeyboardController
import pickle
import socket

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_ip = '127.0.0.1'  # 替换为服务端的IP
server_port = 5556
client_socket.connect((server_ip, server_port))

keyboard_controller = KeyboardController()

# 鼠标点击
def on_click(x, y, button, pressed):
    if pressed:
        message = {"type": "mouse", "action": "click", "x": x, "y": y}
        client_socket.sendall(pickle.dumps(message))

# 键盘按键
def on_press(key):
    try:
        message = {"type": "keyboard", "action": "press", "key": str(key.char)}
    except AttributeError:
        message = {"type": "keyboard", "action": "press", "key": str(key)}
    client_socket.sendall(pickle.dumps(message))

# 启动鼠标和键盘监听
mouse_listener = MouseListener(on_click=on_click)
mouse_listener.start()

keyboard_listener = KeyboardListener(on_press=on_press)
keyboard_listener.start()

while True:
    # 继续接收和显示服务端的屏幕截图
    pass
