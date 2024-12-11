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
        self.group_name_entry.insert(0, "测试")
        self.group_name_entry.grid(row=0, column=1, padx=5, pady=5)

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
        self.user_cache = {}

    def log_message(self, message):
        """在日志窗口中显示信息"""
        self.root.after(0, self._update_log_message, message)

    def _update_log_message(self, message):
        """线程安全地更新日志"""
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.config(state="disabled")
        self.log_text.see(tk.END)

    def start_monitoring(self):
        """启动监听线程"""
        target_keyword = self.group_name_entry.get().strip()

        if not self.is_running:
            self.is_running = True
            self.log_message("启动监听中，请扫码登录...")
            self.monitor_thread = Thread(target=self.run_monitoring, args=(target_keyword, reply_message))
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

    def run_monitoring(self, target_keyword, reply_message):
        """监听逻辑"""
        def on_exit():
            self.log_message("微信已退出登录。可能是被其他端挤下线。自动停止监听。")
            self.stop_monitoring()

        @itchat.msg_register(itchat.content.TEXT, isGroupChat=True)
        def group_text_reply(msg):
            if target_keyword in msg['User']['NickName']:
                # itchat.send(reply_message, toUserName=msg['FromUserName'])  # 回复给该群用户
                # print(f"回复消息给 ：{reply_message}")
                # 如果匹配成功，回复消息给发消息的人
                # 构造要回复的消息
                reply_message = f"@{msg['ActualNickName']}  1"
                # 发送回复消息到群聊，并引用发消息人的内容
                # 这里直接回复给群里的人，并且引用该用户的消息
                itchat.send(reply_message, toUserName=msg['User']['UserName'])  # 回复到群聊中
                self.log_message(f"消息已发送给群聊：{reply_message}")


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
            # 需要使用 after 来确保在主线程更新界面
            self.root.after(0, self._update_qr_code, qr_photo)

    def _update_qr_code(self, qr_photo):
        """线程安全地更新二维码"""
        self.qr_label.config(image=qr_photo)
        self.qr_label.image = qr_photo  # 保存引用，防止被垃圾回收

    def message_match(self, msg_text):
        """匹配消息内容"""
        msg_text = html.unescape(msg_text).replace("\n", " ")
        match = re.search(MESSAGE_PATTERN, msg_text)
        return True

    def get_user_by_id(self, account_id):
        """根据微信ID查找用户"""
        if account_id not in self.user_cache:
            user = itchat.search_friends(userName=account_id)
            self.user_cache[account_id] = user
        return self.user_cache[account_id]

# 启动GUI
root = tk.Tk()
app = WeChatMonitorApp(root)
root.mainloop()
