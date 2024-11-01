# please use Python 3.8.* 版本
import itchat
import html
import re
import tkinter as tk
from tkinter import messagebox
from threading import Thread

# 定义正则表达式模式，适配多行文本
MESSAGE_PATTERN = r"(\d{6}-\d{4}).*?客户需求.*?\+ (\w+) 接单"

class WeChatMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("WeChat Monitor")

        # 定义UI组件
        tk.Label(root, text="群聊名称关键字:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.group_name_entry = tk.Entry(root, width=30)
        self.group_name_entry.insert(0, "技术接单")
        self.group_name_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(root, text="特殊微信ID:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.special_id_entry = tk.Entry(root, width=30)
        self.special_id_entry.insert(0, "tonyhuang2023")
        self.special_id_entry.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(root, text="特殊昵称:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.special_nick_entry = tk.Entry(root, width=30)
        self.special_nick_entry.insert(0, "Tony GenAI")
        self.special_nick_entry.grid(row=2, column=1, padx=5, pady=5)

        self.start_button = tk.Button(root, text="启动监听", command=self.start_monitoring)
        self.start_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.log_text = tk.Text(root, width=50, height=15, state="disabled")
        self.log_text.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

        # 状态变量
        self.is_running = False

    def log_message(self, message):
        """在日志窗口中显示信息"""
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.config(state="disabled")
        self.log_text.see(tk.END)

    def start_monitoring(self):
        """启动监听线程"""
        target_keyword = self.group_name_entry.get().strip()
        special_account_id = self.special_id_entry.get().strip()
        special_nick_name = self.special_nick_entry.get().strip()

        if not target_keyword or not special_account_id or not special_nick_name:
            messagebox.showwarning("输入错误", "请填写所有配置项！")
            return

        if not self.is_running:
            self.is_running = True
            self.log_message("启动监听中，请扫码登录...")
            self.monitor_thread = Thread(target=self.run_monitoring, args=(target_keyword, special_account_id, special_nick_name))
            self.monitor_thread.start()
            self.start_button.config(text="停止监听", command=self.stop_monitoring)
        else:
            self.stop_monitoring()

    def stop_monitoring(self):
        """停止监听"""
        self.is_running = False
        self.log_message("停止监听")
        itchat.logout()
        self.start_button.config(text="启动监听", command=self.start_monitoring)

    def run_monitoring(self, target_keyword, special_account_id, special_nick_name):
        """监听逻辑"""
        @itchat.msg_register(itchat.content.TEXT, isGroupChat=True)
        def group_text_reply(msg):
            # 检查消息是否来自目标群聊（模糊匹配）
            if target_keyword in msg['User']['NickName']:
                match_result = message_match(msg.text)
                if match_result:
                    order_id, account_id = match_result
                    self.log_message(f"匹配成功！订单ID: {order_id}, 微信ID: {account_id}")
                    order_message = f"接单 {order_id}"
                    user = get_user_by_id(account_id, special_account_id, special_nick_name)
                    if user:
                        itchat.send(order_message, toUserName=user[0]['UserName'])
                        self.log_message(f"消息已发送给 {account_id}: {order_message}")
                    else:
                        self.log_message(f"未找到用户 '{account_id}'。")
                else:
                    self.log_message("未找到匹配的消息格式。")

        # 登录并开始监听
        itchat.auto_login(hotReload=True, enableCmdQR=2)
        self.log_message("登录成功！开始监听消息...")
        itchat.run()

def message_match(msg_text):
    msg_text = html.unescape(msg_text).replace("\n", " ")
    match = re.search(MESSAGE_PATTERN, msg_text)
    if match:
        order_id = match.group(1).strip()
        account_id = match.group(2).strip()
        return order_id, account_id
    return None

def get_user_by_id(account_id, special_account_id, special_nick_name):
    if account_id == special_account_id:
        return get_user_by_nickname(special_nick_name)
    user = itchat.search_friends(userName=account_id)
    return user

def get_user_by_nickname(nickname):
    user = itchat.search_friends(nickName=nickname)
    return user

# 启动GUI
root = tk.Tk()
app = WeChatMonitorApp(root)
root.mainloop()