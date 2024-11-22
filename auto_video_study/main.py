import tkinter as tk
from tkinter import ttk


class VideoLearningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("自动学习系统")
        self.root.geometry("800x600")  # 设置窗口大小

        # 设置界面布局
        self.setup_ui()

    def setup_ui(self):
        # 设置上部输入框和登录按钮
        top_frame = tk.Frame(self.root)
        top_frame.pack(side="top", fill="x", pady=10)

        tk.Label(top_frame, text="待登录网站：").pack(side="left", padx=5)
        self.url_entry = tk.Entry(top_frame, width=50)
        self.url_entry.pack(side="left", padx=5)
        login_button = tk.Button(top_frame, text="登录", command=self.login)
        login_button.pack(side="left", padx=5)

        # 设置中间视频列表
        middle_frame = tk.Frame(self.root)
        middle_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        self.video_table = ttk.Treeview(
            middle_frame, columns=("链接", "状态", "播放时长"), show="headings"
        )
        self.video_table.heading("链接", text="视频链接")
        self.video_table.heading("状态", text="状态")
        self.video_table.heading("播放时长", text="播放时长")
        self.video_table.column("链接", width=400, anchor="center")
        self.video_table.column("状态", width=150, anchor="center")
        self.video_table.column("播放时长", width=150, anchor="center")
        self.video_table.pack(fill="both", expand=True)

        # 设置下方日志输出框
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(side="bottom", fill="x", pady=10)

        tk.Label(bottom_frame, text="日志输出：").pack(anchor="w", padx=5)
        self.log_text = tk.Text(bottom_frame, height=8, wrap="word", state="normal")
        self.log_text.pack(fill="both", padx=5, pady=5)

        # 设置底部按钮区域
        button_frame = tk.Frame(self.root)
        button_frame.pack(side="bottom", fill="x", pady=10)

        add_video_button = tk.Button(button_frame, text="添加视频", command=self.add_video)
        add_video_button.pack(side="left", padx=10)
        auto_learn_button = tk.Button(
            button_frame, text="自动学习", command=self.start_learning
        )
        auto_learn_button.pack(side="left", padx=10)

    def login(self):
        """处理登录逻辑"""
        url = self.url_entry.get()
        if not url.strip():
            self.log("请输入网址后再登录！")
            return
        # 模拟登录逻辑
        self.log(f"浏览器打开：{url}")

    def add_video(self):
        """添加视频链接到列表"""
        video_url = self.url_entry.get()
        if not video_url.strip():
            self.log("请输入视频链接后再添加！")
            return
        self.video_table.insert("", "end", values=(video_url, "待学习", "0"))
        self.log(f"添加视频：{video_url}")

    def start_learning(self):
        """模拟启动自动学习"""
        for row in self.video_table.get_children():
            video_url = self.video_table.item(row, "values")[0]
            # 模拟学习逻辑
            status = {"状态": "已完成", "播放时长": "5:00"}
            self.video_table.item(
                row, values=(video_url, status["状态"], status["播放时长"])
            )
            self.log(f"完成学习：{video_url}，状态：{status['状态']}，时长：{status['播放时长']}")

    def log(self, message):
        """记录日志信息"""
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoLearningApp(root)
    root.mainloop()
