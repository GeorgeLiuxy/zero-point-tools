import asyncio
import websockets
import pyautogui
import json
from PIL import Image
import io
import base64

async def handler(websocket, path):
    try:
        while True:
            # 捕获屏幕截图
            screenshot = pyautogui.screenshot()

            # 将截图保存到字节流中
            with io.BytesIO() as output:
                screenshot.save(output, format="PNG")
                img_bytes = output.getvalue()

            # 将图像转换为 base64 编码字符串
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

            # 发送屏幕图像数据给客户端
            await websocket.send(json.dumps({"type": "screen", "data": img_base64}))

            # 尝试接收客户端发送的控制事件
            try:
                event = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                event_data = json.loads(event)

                # 处理鼠标事件
                if event_data["type"] == "mouse":
                    if event_data["action"] == "click":
                        pyautogui.click(event_data["x"], event_data["y"])
                    elif event_data["action"] == "move":
                        pyautogui.moveTo(event_data["x"], event_data["y"])
                    elif event_data["action"] == "scroll":
                        pyautogui.scroll(event_data["dy"], x=event_data["x"], y=event_data["y"])

                # 处理键盘事件
                elif event_data["type"] == "keyboard":
                    key = event_data["key"]
                    if event_data["action"] == "press":
                        pyautogui.keyDown(key)
                    elif event_data["action"] == "release":
                        pyautogui.keyUp(key)

            except asyncio.TimeoutError:
                # 没有接收到新的事件，继续发送屏幕
                pass
            except websockets.ConnectionClosed:
                print("客户端已断开连接")
                break

            # 每隔 100ms 更新一次屏幕
            await asyncio.sleep(0.1)

    except Exception as e:
        print(f"连接错误: {e}")

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("服务器已启动，监听端口 8765")
        await asyncio.Future()  # 运行直到手动停止

if __name__ == "__main__":
    asyncio.run(main())
