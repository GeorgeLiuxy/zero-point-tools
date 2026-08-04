# 桌面窗口远程控制实验

本项目包含两个桌面控制原型：一个是 Flask Web 控制面板，用于查找指定窗口并发送点击/滚动事件；另一个是基于截图、WebSocket 和输入事件的远程桌面实验。

## Flask 控制面板

`app/app.py` 会查找标题中含“投屏”的窗口，保存窗口映射，并通过 `/control` 接口调用 PyAutoGUI 执行点击和滚动。

```bash
pip install flask pyautogui pygetwindow
cd remote_desktop_controller/app
python app.py
```

浏览器访问 `http://localhost:4000`。建议只绑定可信内网，并在实际使用前增加身份认证、CSRF 防护、操作白名单和审计日志。

## WebSocket 演示

- `demo/server.py`：在 `8765` 端口发送屏幕截图并接收鼠标/键盘事件。
- `demo/index.html`：浏览器侧演示页面。
- `demo/client.py`：旧 TCP/pickle 客户端，连接 `5556`，与当前 WebSocket 服务端不兼容。

服务端可能需要 `websockets`、`Pillow`、`pyautogui` 和 `pynput`：

```bash
cd remote_desktop_controller/demo
python server.py
```

## 安全提示

当前原型能够控制真实鼠标和键盘，不应直接暴露到公网。测试前关闭包含密码、隐私或生产系统的窗口，并准备 PyAutoGUI 的紧急停止方式。
