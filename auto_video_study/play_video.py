import tkinter as tk
import vlc
import requests
from tkinter import messagebox

# 创建视频播放窗口
class VideoPlayerApp:
    def __init__(self, root, video_url):
        self.root = root
        self.video_url = video_url

        self.root.title("视频播放与在线学习")
        self.root.geometry("800x600")  # 设置主窗口大小

        # 创建一个Canvas，用于显示视频
        self.canvas = tk.Canvas(self.root, bg='black', width=640, height=360)
        self.canvas.pack(pady=20)

        # 创建按钮控件
        self.play_button = tk.Button(self.root, text="播放视频", command=self.play_video)
        self.play_button.pack()

        # 创建日志显示区
        self.log_text = tk.Text(self.root, width=50, height=10)
        self.log_text.pack(pady=10)

        # 初始化VLC播放器
        self.instance = vlc.Instance('--no-xlib')  # 创建VLC实例
        self.player = self.instance.media_player_new()  # 创建VLC播放器

    def play_video(self):
        """
        播放视频并嵌入到Tkinter的Canvas中
        """
        media = self.instance.media_new(self.video_url)  # 创建媒体对象
        self.player.set_media(media)  # 设置媒体文件
        self.player.set_xwindow(self.canvas.winfo_id())  # 设置Canvas为视频显示区域
        self.player.play()  # 播放视频

        # 播放完成后更新日志
        self.log_text.insert(tk.END, f"正在播放视频：{self.video_url}\n")
        self.root.after(1000, self.check_playing)

    def check_playing(self):
        """
        检查视频是否正在播放，如果播放完成则停止
        """
        if self.player.is_playing():
            self.root.after(1000, self.check_playing)  # 每秒检查一次
        else:
            self.log_text.insert(tk.END, "视频播放完成\n")
            self.play_button.config(state=tk.NORMAL)  # 播放完成后恢复按钮状态

# 获取视频流
def get_video_stream(video_url):
    """
    通过请求获取视频流
    """
    try:
        response = requests.get(video_url, stream=True)
        if response.status_code == 200:
            return response.raw
        else:
            print("视频请求失败")
            return None
    except Exception as e:
        print(f"请求视频流失败: {e}")
        return None

def main():
    video_url = "https://video-230802.ceivs.cn/ftpRoot/course/20230807/20230412091050Z202303300101_1/contents/video_mp4/1.mp4?t=1732788340&k=09e55ce83451666c3a2ec2a2ad21fbe0"

    # 创建主窗口
    root = tk.Tk()

    # 创建视频播放器应用
    app = VideoPlayerApp(root, video_url)

    # 启动GUI
    root.mainloop()

# 启动程序
if __name__ == "__main__":
    main()
