
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

class BrowserManager:
    def __init__(self):
        # 指定 ChromeDriver 路径
        # 定义 ChromeOptions
        options = webdriver.ChromeOptions()
        service = Service("/Volumes/software/chromedriver-mac-arm64/chromedriver")
        self.driver = webdriver.Chrome(service=service, options=options)

    def open_url(self, url):
        """打开指定的 URL"""
        self.driver.get(url)

    def find_element(self, by, value):
        """封装的查找元素方法"""
        try:
            return self.driver.find_element(by, value)
        except:
            return None

    def click_button(self, by, value):
        """点击按钮"""
        button = self.find_element(by, value)
        if button:
            button.click()

    def play_video(self):
        """自动播放视频"""
        video = self.find_element(By.TAG_NAME, "video")
        if video:
            self.driver.execute_script("arguments[0].play();", video)

    def stop_video(self):
        """停止视频播放"""
        video = self.find_element(By.TAG_NAME, "video")
        if video:
            self.driver.execute_script("arguments[0].pause();", video)

    def close(self):
        """关闭浏览器"""
        self.driver.quit()
