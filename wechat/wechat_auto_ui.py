# please use Python 3.8.* 版本
import itchat
import html
import re
import tkinter as tk
from tkinter import messagebox
from threading import Thread
from PIL import Image, ImageTk
import io

# 定义正则表达式模式，适配多行文本
MESSAGE_PATTERN = r"(\d{6}-\d{4}).*?客户需求.*?\+ (\w+) xxx"

class WeChatMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("WeChat Monitor")

        # 定义UI组件
        tk.Label(root, text="群聊名称关键字:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.group_name_entry = tk.Entry(root, width=30)
        self.group_name_entry.insert(0, "技术xxx")
        self.group_name_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(root, text="特殊微信ID:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.special_id_entry = tk.Entry(root, width=30)
        self.special_id_entry.insert(0, "xxxxx")
        self.special_id_entry.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(root, text="特殊昵称:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.special_nick_entry = tk.Entry(root, width=30)
        self.special_nick_entry.insert(0, "xxxx")
        self.special_nick_entry.grid(row=2, column=1, padx=5, pady=5)

        self.start_button = tk.Button(root, text="启动监听", command=self.start_monitoring)
        self.start_button.grid(row=3, column=0, columnspan=2, pady=10)

        # 创建用于显示二维码的标签
        self.qr_label = tk.Label(root)
        self.qr_label.grid(row=4, column=0, columnspan=2, pady=5)

        # 创建日志显示窗口和滚动条
        self.log_frame = tk.Frame(root)
        self.log_frame.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

        self.log_text = tk.Text(self.log_frame, width=50, height=15, state="disabled", wrap="word")
        self.log_scrollbar = tk.Scrollbar(self.log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=self.log_scrollbar.set)

        self.log_text.pack(side="left", fill="both", expand=True)
        self.log_scrollbar.pack(side="right", fill="y")

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
        if self.is_running:
            self.is_running = False
            self.log_message("停止监听")
            itchat.logout()
            self.start_button.config(text="启动监听", command=self.start_monitoring)
            self.qr_label.config(image="")  # 清除二维码

    def run_monitoring(self, target_keyword, special_account_id, special_nick_name):
        """监听逻辑"""

        def on_exit():
            self.log_message("微信已退出登录。可能是被其他端挤下线。自动停止监听。")
            self.stop_monitoring()

        @itchat.msg_register(itchat.content.TEXT, isGroupChat=True)
        def group_text_reply(msg):
            if target_keyword in msg['User']['NickName']:
                match_result = message_match(msg.text)
                if match_result:
                    order_id, account_id = match_result
                    self.log_message(f"匹配成功！订单ID: {order_id}, 微信ID: {account_id}")
                    order_message = f"xxx {order_id}"
                    user = get_user_by_id(account_id, special_account_id, special_nick_name)
                    if user:
                        itchat.send(order_message, toUserName=user[0]['UserName'])
                        self.log_message(f"消息已发送给 {account_id}: {order_message}")
                    else:
                        self.log_message(f"未找到用户 '{account_id}'。")
                else:
                    self.log_message("未找到匹配的消息格式。")

        # 登录并开始监听，使用二维码回调来显示二维码
        itchat.auto_login(
            hotReload=True,
            enableCmdQR=False,
            qrCallback=self.show_qr_code,  # 设置二维码回调
            exitCallback=on_exit
        )
        self.log_message("登录成功！开始监听消息...")
        self.qr_label.config(image="")  # 登录成功后隐藏二维码

        try:
            itchat.run()
        except Exception as e:
            self.log_message(f"监听过程中出现异常: {e}")
            self.stop_monitoring()

    def show_qr_code(self, uuid=None, status=0, qrcode=None):
        """显示二维码到UI中"""
        if qrcode:  # 确保二维码数据存在
            qr_image = Image.open(io.BytesIO(qrcode))
            qr_photo = ImageTk.PhotoImage(qr_image)
            self.qr_label.config(image=qr_photo)
            self.qr_label.image = qr_photo  # 保存引用，防止被垃圾回收

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