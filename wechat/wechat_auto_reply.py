import applescript
import time
import schedule
import subprocess

# 指定的微信群名称
GROUP_NAME = "测试"
# 自动回复的内容
REPLY_MESSAGE = "自动回复：收到您的消息！"

def open_group_chat():
    """打开微信指定的群聊窗口"""
    script = f'''
    tell application "WeChat"
        activate
        delay 1
        tell application "System Events"
            keystroke "F" using {{command down, shift down}}
            delay 0.5
            keystroke "{GROUP_NAME}"
            delay 1
            keystroke return
            delay 1
            keystroke "A" using {{command down}}
            delay 0.5
            keystroke "C" using {{command down}}
            delay 0.5
        end tell
    end tell
    '''
    print("运行 AppleScript 以打开微信群并复制消息内容...")
    applescript.run(script)

def get_latest_messages():
    """从剪贴板获取最近的消息内容"""
    result = subprocess.run(['pbpaste'], stdout=subprocess.PIPE)
    messages = result.stdout.decode('utf-8')
    print("获取到的消息内容为：", messages)
    return messages

def send_reply():
    """发送自动回复消息"""
    script = f'''
    tell application "WeChat"
        activate
        delay 1
        tell application "System Events"
            keystroke "{REPLY_MESSAGE}"
            delay 0.5
            keystroke return
        end tell
    end tell
    '''
    print("发送自动回复消息...")
    applescript.run(script)

def check_and_reply():
    """检查微信群消息，如果符合条件则自动回复"""
    open_group_chat()
    time.sleep(1)  # 等待消息加载
    messages = get_latest_messages()

    # 检查是否包含特定关键词
    if "特定关键词" in messages:  # 修改为实际检测的关键词
        print("检测到符合条件的消息，发送自动回复...")
        send_reply()
    else:
        print("未检测到符合条件的消息")

# 设置每隔 1 分钟检查一次微信群消息
schedule.every(1).minutes.do(check_and_reply)

print("微信自动回复脚本已启动...")
while True:
    schedule.run_pending()
    time.sleep(1)