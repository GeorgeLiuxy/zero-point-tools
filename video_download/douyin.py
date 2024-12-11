import json
import re
import uuid
from pprint import pprint
from urllib.parse import unquote

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
        "cookie": (
            "douyin.com; xg_device_score=7.407134386430824; device_web_cpu_core=8; device_web_memory_size=8; ttwid=1%7CJ2V-Tj5noOIl6NXdt4hvojkblBLBbkUSx3bYigWNyk0%7C1727425195%7Cef44d598365c8c2cd18682fceaf20163c702524fa2d5b84956d356e794baa925; UIFID_TEMP=9e5c45806baed1121aef2e4ebdb50ae0783a7b9267143d29acaade7dde1bacd576b10274108da1a5262d729836cdae2445052212d4bfe4dfebdd6f466b3bbd9f1b8877b896f47a776f52f4a3dc1af6d0; hevc_supported=true; fpk1=U2FsdGVkX19YtIvcyNEI/vNTruTK4dk7Mk1tPpTUaFF3RIW+PAZZAsOdNv9aQWs0uSwvKq/JxYDqjfJvVllE9Q==; fpk2=fe0673f2a48d047b912b27e2a0c02f9f; xgplayer_user_id=208981754985; UIFID=9e5c45806baed1121aef2e4ebdb50ae0783a7b9267143d29acaade7dde1bacd56670f279b054b34a168e36104d5e68d46fdd7634ad33f55d20f1c2fe50de78c5ef4dfb7915f3f6b7b53f8eaf45ecdb29f69df1e410430ccc34ee6b829b49f386531b4cad7ec502f217ad06e0faba75bbfdc1378bb7fcb96483f559601aef7b0b1049426fce47f244e1e83a2469d825d11c1bb55f3f17444459f8f0b3ba3d9253; odin_tt=10da37992fc73c01c07135f01f20ceb9cde94597f673f2526785a73d0cfb18c7dde4bf92b185ea10a72bb72b76444c7ed71a3a0a301302bde0484278e98db01a853c30fb15f4b853c3cccaf801de3d31; bd_ticket_guard_client_web_domain=2; d_ticket=99217e3f2204cf30161082e1d3d8d1aa9c866; volume_info=%7B%22isUserMute%22%3Afalse%2C%22isMute%22%3Atrue%2C%22volume%22%3A0.5%7D; SEARCH_RESULT_LIST_TYPE=%22single%22; s_v_web_id=verify_m483zf70_co99h7em_yCgw_4jcZ_89O3_S3vDH9nsLJeW; dy_swidth=1680; dy_sheight=1050; csrf_session_id=0d8e8fa9dcfe5d62c7a31c3a1945a2b8; is_dash_user=1; passport_csrf_token=f9cb74d6f4e2fb516d1dbc3281e4b839; passport_csrf_token_default=f9cb74d6f4e2fb516d1dbc3281e4b839; FORCE_LOGIN=%7B%22videoConsumedRemainSeconds%22%3A180%7D; __ac_nonce=06751b57800aa89e93c61; __ac_signature=_02B4Z6wo00f012xBCBQAAIDAPr4oDMpOno9sYQyAALxN91; stream_recommend_feed_params=%22%7B%5C%22cookie_enabled%5C%22%3Atrue%2C%5C%22screen_width%5C%22%3A1680%2C%5C%22screen_height%5C%22%3A1050%2C%5C%22browser_online%5C%22%3Atrue%2C%5C%22cpu_core_num%5C%22%3A8%2C%5C%22device_memory%5C%22%3A8%2C%5C%22downlink%5C%22%3A10%2C%5C%22effective_type%5C%22%3A%5C%224g%5C%22%2C%5C%22round_trip_time%5C%22%3A100%7D%22; strategyABtestKey=%221733408122.229%22; home_can_add_dy_2_desktop=%221%22; biz_trace_id=ed84b988; bd_ticket_guard_client_data=eyJiZC10aWNrZXQtZ3VhcmQtdmVyc2lvbiI6MiwiYmQtdGlja2V0LWd1YXJkLWl0ZXJhdGlvbi12ZXJzaW9uIjoxLCJiZC10aWNrZXQtZ3VhcmQtcmVlLXB1YmxpYy1rZXkiOiJCTCt1cVRPdEdxT1RERjZkK2daZkhtMTJCczdBSWhPT1hRQW95OWlBVHRIQytPS0IzbDJlck4xTTNDTXc0NWl0Mmo1MWtma3gwblU0U05uaC9HaW1uNGM9IiwiYmQtdGlja2V0LWd1YXJkLXdlYi12ZXJzaW9uIjoyfQ%3D%3D; sdk_source_info=7e276470716a68645a606960273f276364697660272927676c715a6d6069756077273f276364697660272927666d776a68605a607d71606b766c6a6b5a7666776c7571273f275e5927666d776a686028607d71606b766c6a6b3f2a2a61676f67606875696f6d66686d69637563646664696a686a6b6f756469756e6a2a7666776c7571762a6c6b76756066716a772b6f76592758272927666a6b766a69605a696c6061273f27636469766027292762696a6764695a7364776c6467696076273f275e5827292771273f273133343037343d3531363632342778; bit_env=VQBKjirWxNMB1OfN4nfhDrPCUmtK8_z_z6TklUt_7o0NevOzuVAu3DWP3YwFb3qG2CWs2inQg6yAFPSM7PCNAw5rlgwbWLjD7DB2DpNeB6KsgzsBbhLkxRXtC_N6ozEyt7uaL9p-AMvIivr6yXpPdMJCy6hIY95pvQtPT32sRoGEU3HOVmE3Js1CD5ef-q1D8ow9VYbpnNd_ls_a7BSHd84P0l9U8Mry1fAfohuPXSWFAcTyrxb9-o65QBU22QYvluVH-V97lhpuNv0-v3AzEUBg0OgG_nu1MZPiUyFd_mBg8b_Fy4INVsHPuiXn4F0yXKPPST68HcNvED8QyEsi6a9ZKr2Rw6igmgZfS0DYGHqzQi2Tl_RZsxufeD_dMJeeFPkjcxHyXSY2WIPesphU4kmDDoAeJum8cdW9Ha-u4elmy7TFwBfk4JyGTOwWRf9OEMIUTagCFt2zL1_gq08i90ZRcWCffRsgNTyt6PYFklKXAIUMPV3ANJqNlLtO93LywNQ1zsrrYE3PvazGhZ7AWPuOEz1qy7CmQLfwelDTWjk%3D; gulu_source_res=eyJwX2luIjoiMDhjOGQ3ZTJiODQyNjZkZWI5Y2VkMGJiODNlNmY1ZWY0ZjMyNTE2ZmYyZjAzNDMzZjI0OWU1Y2Q1NTczNTk5NyJ9; passport_auth_mix_state=qworxxzba7b10iakre9zcj1ef2ue6rgj; download_guide=%222%2F20241205%2F0%22; IsDouyinActive=false"
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
        info = re.findall('self.__pace_f.push\(\[1,"(.*?)"\]\)</script>', html_content)[0]
        info2 = re.findall('<script id="RENDER_DATA" type="application/json">(.*?)</script>', html_content)[0]
        json_data = json.loads(unquote(info2))
        # pprint(json_data)
        print(response.cookies.get_dict())

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
    video_url = 'https://www.douyin.com/video/7280113311302176034'
    download_b_site_video(video_url, video_url)
