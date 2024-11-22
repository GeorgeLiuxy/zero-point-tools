import time

class VideoManager:
    def __init__(self, browser_manager):
        self.browser_manager = browser_manager
        self.video_list = {}

    def add_video(self, video_url):
        """添加视频链接"""
        self.video_list[video_url] = {"状态": "待学习", "播放时长": "0"}

    def start_learning(self, video_url):
        """开始学习视频"""
        self.browser_manager.open_url(video_url)

        # 点击“开始学习”按钮（如果有）
        self.browser_manager.click_button("xpath", "//button[text()='开始学习']")

        # 播放视频
        self.browser_manager.play_video()
        start_time = time.time()

        # 等待视频播放完成（可调整为实际播放检测逻辑）
        time.sleep(10)  # 模拟 10 秒播放时间

        # 停止视频播放并点击“结束学习”按钮（如果有）
        self.browser_manager.stop_video()
        self.browser_manager.click_button("xpath", "//button[text()='结束学习']")

        # 更新播放时长
        duration = int(time.time() - start_time)
        self.video_list[video_url]["status"] = "已完成"
        self.video_list[video_url]["duration"] = duration

    def get_video_duration(self, video_url):
        """获取视频播放时长"""
        return self.video_list[video_url]["duration"]
