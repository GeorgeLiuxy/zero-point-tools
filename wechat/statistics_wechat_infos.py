import itchat
from itchat.content import TEXT, SHARING

def log_message(msg_type, msg_content, sender):
    """打印消息的内容"""
    print(f"[{msg_type}] 来自 {sender}: {msg_content}")

@itchat.msg_register([TEXT, SHARING], isFriendChat=True, isGroupChat=True, isMpChat=True)
def handle_message(msg):
    """处理收到的消息"""
    msg_type = msg['Type']
    msg_content = msg['Text']
    sender = msg['User']['NickName'] if 'NickName' in msg['User'] else "未知"
    log_message(msg_type, msg_content, sender)

@itchat.msg_register([TEXT], isFriendChat=True, isGroupChat=True, isMpChat=True)
def handle_sent_message(msg):
    """处理发出的消息"""
    msg_type = msg['Type']
    msg_content = msg['Text']
    sender = "我自己"
    log_message(msg_type, msg_content, sender)

def main():
    """主函数，登录并启动监听"""
    # 登录微信，启用热登录避免频繁扫码
    itchat.auto_login(hotReload=True)
    print("登录成功！开始监听所有消息...")

    # 开始运行消息监听
    itchat.run()

if __name__ == "__main__":
    main()
