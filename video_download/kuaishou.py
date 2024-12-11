import json
import re

import requests
import yt_dlp

'''
实例实现步骤
    一、数据来源分析（要有当你找到了数据来源的时候，才能够通过代码实现）
        1、爬取用户下对应的视频，保存mp
        2、通过开发者工具进行抓包分析，分析数据从哪里来的（找出真正的数据源）
            动态加载页面， 开发者工具抓包数据
    二、代码实现过程
        1、找到目标网址
        2、发送请求
            get、post
        3、解析数据（获取视频地址，视频标题）
        4、发送请求 请求每个视频地址
        5、保存视频
'''

def download_kuaishou_video(url, referer):
    """
    下载快手视频。
    """
    import requests

    # 设置请求头部
    headers = {
        "content-type": "application/json",
        "Cookie": "weblogger_did=web_86104148745314AD; kpf=PC_WEB; clientid=3; did=web_7aa971004a70715dab30c2a6a5f34b8e; kpn=KUAISHOU_VISION",
        "Host": "www.kuaishou.com",
        "Origin": "https://www.kuaishou.com",
        "Referer": referer,
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    }
    # 请求的 URL 获取视频链接
    response = requests.post(url, headers=headers, timeout=10)
    if response.status_code != 200:
        print(f"请求失败，状态码: {response.status_code}")
        return
    else:
        # 假设页面的 HTML 内容存储在 `html_content` 变量中
        html_content = response.content
        html_content = html_content.decode('utf-8')  # 或使用适当的编码
        # 匹配 <script> 标签中的内容
        script_content = re.search(r'<script>(.*?)</script>', html_content, re.DOTALL)
        if script_content:
            apollo_state = re.search(r'window\.__APOLLO_STATE__\s*=\s*(\{.*?\});', script_content.group(1), re.DOTALL)
            if apollo_state:
                json_content = apollo_state.group(1)
                if json_content:
                    # 将 JSON 字符串解析为字典
                    json_content = json.loads(json_content)
                    cover_url = json_content['defaultClient']['VisionVideoDetailPhoto:3xjfnb6vp5thbfu']['photoUrl']
                    if cover_url:
                        save_video(cover_url)
                    else:
                        print("没有找到视频地址")
            else:
                print("没有找到 Apollo State")
        else:
            print("没有找到 <script> 标签")
        # print(video_url)

def save_video(video_url, output_file="xx.mp4"):
    # 发送 GET 请求获取视频内容
    video_data = requests.get(video_url).content

    # 保存视频内容到 MP4 文件
    with open(output_file, 'wb') as file:
        file.write(video_data)

    print(f"视频已成功保存为 {output_file}")


if __name__ == "__main__":
    # 示例快手视频链接
    video_url = 'https://www.kuaishou.com/short-video/3xjfnb6vp5thbfu'
    download_kuaishou_video(video_url, video_url)
