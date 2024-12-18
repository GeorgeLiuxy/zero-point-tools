import pygetwindow as gw
import pyautogui

# 控制窗口的操作
def control_window(window_title, action, x=None, y=None, delta=None):
    # 获取窗口
    window = gw.getWindowsWithTitle(window_title)
    if not window:
        return {"error": "Window not found"}

    window = window[0]

    # 控制点击
    if action == "click":
        pyautogui.click(x, y)
    # 控制滚动
    elif action == "scroll" and delta is not None:
        pyautogui.scroll(delta)
    else:
        return {"error": "Invalid action"}

    return {"status": "success"}
