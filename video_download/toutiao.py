import os
import uuid
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

# 配置Chrome选项
chrome_options = Options()
chrome_options.add_argument('--headless')  # 启用无头模式（后台运行）
chrome_options.add_argument('--disable-gpu')  # 禁用GPU硬件加速
chrome_options.add_argument('accept-language=zh-CN,zh;q=0.9')  # 设置语言偏好
chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36')

# 指定ChromeDriver的路径
driver_path = 'chromedrivers/chromedriver-mac-arm64/chromedriver'  # 请替换为你自己的chromedriver路径
service = Service(driver_path)

# 启动WebDriver
driver = webdriver.Chrome(service=service, options=chrome_options)

# 访问目标URL
url = 'https://www.ixigua.com/embed?group_id=7440454339035922954'
driver.get(url)

# 等待直到 <video> 元素加载完成
try:
    video_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, 'video'))
    )
    # 获取视频的 URL
    video_url = video_element.get_attribute('src')

    # 处理相对路径
    if video_url.startswith("//"):
        video_url = "https:" + video_url

    print(f"视频的 URL: {video_url}")

    # 获取当前页面的 cookies
    cookies = driver.get_cookies()
    cookie_dict = {cookie['name']: cookie['value'] for cookie in cookies}

    # 设置请求头
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Referer": "https://www.ixigua.com/7436667895561585190",
        "accept": "application/json, text/plain, */*",
        "pragma": "no-cache",
        "cache-control": "no-cache",
        "sec-ch-ua-platform": "macOS",
        "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-fetch-site": "same-origin",
        "sec-fetch-mode": "cors",
        "sec-fetch-dest": "empty",
        "accept-encoding": "gzip, deflate, br, zstd",
        "x-secsdk-csrf-token": "0001000000015fa39beec3aaa3d025fc3f2864cc78e4f2c1a8fddff026bbdb955bdf7536c425180e4be473a23e27",
    }

    # 手动设置 cookies
    session = requests.Session()
    for cookie_name, cookie_value in cookie_dict.items():
        session.cookies.set(cookie_name, cookie_value)

    # 获取视频文件的大小（HEAD 请求）
    head_response = session.head(video_url, headers=headers)

    # 获取文件大小（单位：字节）
    content_length = int(head_response.headers.get('Content-Length', 0))
    print(f"视频文件大小: {content_length / (1024 * 1024):.2f} MB")

    # 判断视频是否大于 300MB
    if content_length > 300 * 1024 * 1024:
        print("视频文件太大，超过300MB，跳过下载")
    else:
        # 使用 requests 下载视频，并携带 cookies 和 headers
        response = session.get(video_url, headers=headers, stream=True)

        if response.status_code == 200:
            # 生成随机UUID文件名
            file_name = str(uuid.uuid4()) + ".mp4"
            file_path = os.path.join("downloads", file_name)  # 视频保存路径

            # 创建保存目录（如果不存在）
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 保存视频文件
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)

            print(f"视频已保存: {file_path}")
        else:
            print(f"请求失败，状态码: {response.status_code}")

except Exception as e:
    print(f"发生错误: {e}")

# 关闭 WebDriver
driver.quit()
