import json
import re
import uuid
from pprint import pprint

import requests
import yt_dlp
import subprocess
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

def download_b_site_video(url, referer):
    """
    下载快手视频。
    """
    import requests

    # 设置请求头部
    headers = {
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "origin": "https://www.bilibili.com",
        "referer": referer,
        "cookie": (
            "buvid3=6F4EC48C-25EC-F0E8-ECF2-CFEFA967419921441infoc; "
            "b_nut=1726275221; _uuid=12B84119-52F3-6948-B55F-7BAF8106117A622360infoc; "
            "CURRENT_FNVAL=4048; buvid_fp=8759d3c452c4c0d739b52e489fe21f18; "
            "buvid4=198DDF34-1FFD-9220-0782-55D81D09864522917-024091400-OcCVRwXq4joxt2wQAQ8I1Q%3D%3D; "
            "rpdid=|(kJYYJlJkml0J'u~kYk~lkkJ; header_theme_version=CLOSE; "
            "enable_web_push=DISABLE; home_feed_column=4; browser_resolution=1354-470; "
            "DedeUserID=1262554758; DedeUserID__ckMd5=03d5eac81bb0b7bf; "
            "bsource=search_baidu; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MzM1NjM0NjIsImlhdCI6MTczMzMwNDIwMiwicGx0IjotMX0.rmx5ip-NffFD38RIYNatlmWuUsPhBFXhvn8WSIFPozI; "
            "bili_ticket_expires=1733563402; SESSDATA=2bd78ed7%2C1748856266%2Cb94f1%2Ac2CjCGE14m5G5WJ6BlpAiEGXlIYQw0uSDUOZfpiUrAtt4zXX2jRr-x74NezVaP01I5FbESVnZtT18wOFFBdFU3dlNBRmE5Yi1OV1gzY0JQLURKZWUyblJFUk1XeDdDQlhzeXZxMlBTTG9ZRFlIR29vOHg5QndYb3F4SzE3cEZvWlZaZDVSbUt2X2dnIIEC; "
            "bili_jct=08992dadea63b20dd2b1d412750f36cd; sid=4si4jxlf; CURRENT_QUALITY=16; "
            "bp_t_offset_1262554758=1007025992457256960; b_lsid=DD10256B9_19391597857"
        ),
        "priority": "u=1, i"
    }

    # 请求的 URL 获取视频链接
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code != 200:
        print(f"请求失败，状态码: {response.status_code}")
        return
    else:
        # 假设页面的 HTML 内容存储在 `html_content` 变量中
        html_content = response.text
        # html_content = html_content.decode('utf-8')  # 或使用适当的编码

        html_content = html_content.replace("' \n '", '')
        # 匹配 <script> 标签中的内容
        script_content = re.findall('<script>window.__playinfo__=(.*?)</script>', html_content)
        if script_content:
            # 获取第一个匹配项，假设它是一个 JSON 字符串
            script_content = script_content[0]
            json_obj = json.loads(script_content)
            if json_obj:
                audio_url = json_obj['data']['dash']['audio'][0]['baseUrl']
                video_url = json_obj['data']['dash']['video'][0]['baseUrl']
                if audio_url and video_url:
                    save_video_audio(video_url, audio_url)
                else:
                    print("没有找到视频地址")
            else:
                print("没有找到 Apollo State")
        else:
            print("没有找到 <script> 标签")
        # print(video_url)

def save_video_audio(video_url, audio_url):
    # 生成一个随机的 UUID
    random_uuid = uuid.uuid4()
    video_name = f"video_{random_uuid}.mp4"
    audio_name = f"video_{random_uuid}.mp3"
    video_data = requests.get(video_url).content
    audio_data = requests.get(audio_url).content
    print(video_url)
    print(audio_url)
    # 保存视频内容到 MP4 文件
    with open(video_name, 'wb') as file:
        file.write(video_data)
    # 保存视频内容到 MP4 文件
    with open(audio_name, 'wb') as file:
        file.write(audio_data)

    # merge_audio_video(video_name, audio_name, 'output_video.mp4')

    print(f"视频已成功保存为 {video_name}")


def merge_audio_video(video_file, audio_file, output_file):
    command = [
        'ffmpeg',
        '-i', video_file,  # 输入的视频文件
        '-i', audio_file,  # 输入的音频文件
        '-c:v', 'copy',     # 拷贝视频编码
        '-c:a', 'aac',      # 使用 AAC 音频编码
        '-strict', 'experimental',  # 允许使用实验性编码器
        output_file         # 输出文件
    ]

    subprocess.run(command)


if __name__ == "__main__":
    # 示例快手视频链接
    video_url = 'https://www.bilibili.com/video/av113530800053067/'
    download_b_site_video(video_url, video_url)
