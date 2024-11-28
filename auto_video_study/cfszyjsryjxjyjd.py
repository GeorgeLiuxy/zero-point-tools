import base64
import datetime
import hashlib
import queue
import sys
import threading
import time
import tkinter as tk
from io import BytesIO
from tkinter import messagebox

import requests
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from PIL import Image
import pytesseract

''' 赤峰市专业技术人员继续教育基地'''

START_DATE = datetime.date(2024, 11, 28)  # 示例日期，你可以选择程序首次运行的日期

# 登录API URL
login_url = "https://portalapi.tlsjyy.com.cn/api/login/login"
# kaptcha_url = f"https://gp.chinahrt.com/gp6/system/manager/kaptcha?d={timestamp_str}"

def md5_encrypt(password):
    # 使用 MD5 对密码进行加密
    md5 = hashlib.md5()
    md5.update(password.encode('utf-8'))  # 将密码编码为字节流
    return md5.hexdigest()  # 返回加密后的十六进制字符串


def login(username, password):
    global headers
    data = {
        "username": username,
        "password": md5_encrypt(password)
    }
    # 发送POST请求进行登录
    login_response = requests.post(login_url, data=data, timeout=10)
    # 打印返回的响应内容（通常会是JSON格式）
    if login_response.status_code == 200:
        # 解析返回的JSON响应
        login_data = login_response.json()
        data = login_data['data']
        print("登录成功，返回数据:", login_data)
        # 设置请求头，模拟浏览器行为
        headers = {
            "pragma": "no-cache",
            "cache-control": "no-cache",
            "sec-ch-ua-platform": "macOS",
            "authorization": data['access_token'],
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "accept": "application/json, text/plain, */*",
            "sec-ch-ua": "Google Chrome;v=131, Chromium;v=131, Not_A Brand;v=24",
            "content-type": "application/json;charset=UTF-8",
            "sec-ch-ua-mobile": "?0",
            "origin": "https://peixun.tlsjyy.com.cn",
            "sec-fetch-site": "same-site",
            "sec-fetch-mode": "cors",
            "sec-fetch-dest": "empty",
            "referer": "https://peixun.tlsjyy.com.cn/",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "zh-CN,zh;q=0.9",
            "priority": "u=1, i"
        }


    else:
        print("登录失败，状态码:", login_response.status_code)
    return headers


def getStudyVideo(chapter, train_id, check_study_url):
    global response
    chapter_id = chapter['id']
    # 调用check_study接口
    check_study_params = {
        'chapter_id': chapter_id,
        'train_id': train_id
    }
    # 发送GET请求检查学习状态
    response = requests.get(check_study_url, headers=headers, params=check_study_params, timeout=10)
    study_params = {
        'chapter_id': chapter_id,
        'train_id': train_id
    }
    study_url = f"https://peixunapi.tlsjyy.com.cn/api/course/study"
    if response.status_code == 200:
        check_study_result = response.json()
        # 判断响应数据类型
        if check_study_result['data']:
            # 有数据，则点击同意跳转到指定课程页面继续学习
            print(f"章节 {chapter['title']} 学习状态: 正在学习，详情: {check_study_result['data']}")
            study_params['chapter_id'] = check_study_result['data']['id']
        else:
            # 无数据，直接进行改章节的学习
            print(f"章节 {chapter['title']} 学习状态: 未学习或学习信息缺失")
            study_params['chapter_id'] = chapter_id
    else:
        print(f"检查章节 {chapter['title']} 学习状态失败，状态码: {response.status_code}")
    study_response = requests.get(study_url, headers=headers, params=study_params, timeout=10)
    if study_response.status_code == 200:
        study_info = study_response.json()
        print(f"章节 {chapter['title']} 学习开始成功，开始学习...{study_info}")
    else:
        print(f"章节 {chapter['title']} 学习开始失败，状态码: {study_response.status_code}")
    return study_info


# AES加密函数
def aes_encrypt(plain_text, key):
    cipher = AES.new(key.encode(), AES.MODE_ECB)  # 使用ECB模式
    padded_text = pad(plain_text.encode(), AES.block_size)  # 填充明文
    encrypted_text = cipher.encrypt(padded_text)  # 加密
    return base64.b64encode(encrypted_text).decode('utf-8')  # 返回base64编码的密文


# 获取视频播放进度并调用接口更新学习状态
def end_study(headers, chapter_id, train_id, video_url, time_length, study_time_length, log_queue):
    # 模拟视频播放
    video_duration = time_length  # 假设视频的总时长，单位秒
    current_time = study_time_length  # 假设当前播放时间
    # log_queue.put(f"开始播放视频：{video_url}")
    while current_time < video_duration:
        # 假设每秒钟播放1秒，模拟视频播放
        time.sleep(1)
        current_time += 1
        # log_queue.put(f"当前播放进度：{current_time}/{video_duration}秒")

    log_queue.put(f"播放视频结束：{video_url}")
    # 视频播放结束时，调用end_study接口
    time_length = aes_encrypt(str(current_time), "stu_card_1847521")  # 加密视频播放时间
    # 调用结束学习接口
    url = "https://peixunapi.tlsjyy.com.cn/api/course/end_study"
    data = {
        "chapter_id": chapter_id,
        "train_id": train_id,
        "time_length": time_length
    }
    response = requests.post(url, json=data, headers=headers, timeout=10)
    if response.status_code == 200:
        log_queue.put(f"章节{chapter_id} 完成学习状态更新成功：{video_url}")
    else:
        print(f"学习状态更新失败，状态码: {response.status_code}")


def start_study(headers, course_params, log_queue):
    try:
        all_chapters = []
        all_chapters.extend(getCourseAndChapters(headers, course_params[0]))  # 加载 bx 课程
        all_chapters.extend(getCourseAndChapters(headers, course_params[1]))  # 加载 xx 课程

        log_queue.put(f'有 {len(all_chapters)} 个章节需要学习\n')

        check_study_url = "https://peixunapi.tlsjyy.com.cn/api/course/check_study"
        train_id = 1701
        log_queue.put(f'你还有 {len(all_chapters)} 个章节需要')
        for chapter in all_chapters:
            study_info = getStudyVideo(chapter, train_id, check_study_url)
            log_queue.put(f"课程 {chapter['course']['title']}章节 {chapter['title']}")
            time_length = study_info['data']['time_length']
            study_time_length = study_info['data']['study_time_length']
            chapter_id = study_info['data']['id']
            for video_url in study_info['data']['video_url']:
                log_queue.put(f"正在观看视频: {video_url}\n")
                # 执行
                end_study(headers, chapter_id, train_id, video_url, time_length, study_time_length,log_queue)
                time.sleep(1)  # 模拟观看视频的延时

        log_queue.put("学习结束\n")
    except Exception as e:
        messagebox.showerror("错误", str(e))


def check_trial_period():
    current_date = datetime.date.today()  # 获取当前日期
    # 计算当前日期与开始日期的差值
    date_diff = (current_date - START_DATE).days

    if date_diff > 30:
        return True  # 超过30天，表示试用期已过
    else:
        return False  # 试用期未过


def on_login_button_click(username_entry, password_entry, login_button, log_queue):
    # 检查试用期是否过期
    if check_trial_period():
        messagebox.showerror("试用期已过期", "您的试用期已过期，程序无法继续使用。请联系管理员")
        sys.exit()  # 程序退出
    else:
        username = username_entry.get()
        password = password_entry.get()
        try:
            # 禁用登录按钮，避免重复点击
            login_button.config(state=tk.DISABLED)
            # 创建新线程进行登录操作

            threading.Thread(target=login_study, args=(username, password, log_queue), daemon=True).start()  # 启动后台线程
        except Exception as e:
            log_queue.put(f"登录失败: {e}\n")


# 定期检查队列中的消息并更新 UI
def update_log(result_text_widget, log_queue):
    try:
        while True:
            log_message = log_queue.get_nowait()  # 非阻塞获取消息
            result_text_widget.insert(tk.END, log_message)
            result_text_widget.yview(tk.END)  # 滚动到最后一行
    except queue.Empty:
        pass
    result_text_widget.after(100, update_log, result_text_widget, log_queue)  # 每100ms检查一次


def get_current_time_stamp():
    # 获取当前的日期和时间
    now = datetime.now()
    # 获取时间戳（秒级），并转换为毫秒
    timestamp_milliseconds = int(now.timestamp() * 1000)
    print(f"毫秒级时间戳: {timestamp_milliseconds}")
    return timestamp_milliseconds



def get_captcha_image(kaptcha_url):
    # 发送请求获取验证码图片
    response = requests.get(kaptcha_url)
    if response.status_code == 200:
        # 保存验证码图片
        with open('captcha.jpg', 'wb') as f:
            f.write(response.content)
        print("验证码图片已保存！")
        return 'captcha.jpg'
    else:
        print(f"获取验证码失败，HTTP状态码: {response.status_code}")
        return None

def parse_captcha(captcha_image_path):
    # 载入验证码图片
    captcha_image = Image.open(captcha_image_path)

    # 使用 Tesseract 解析验证码
    captcha_text = pytesseract.image_to_string(captcha_image).strip()

    # 输出解析结果
    print(f"解析出的验证码是：{captcha_text}")
    return captcha_text


# 验证验证码
def valid(captcha_text, username, password):
    valid_url = "https://gp.chinahrt.com/gp6/system/manager/login/valid"
    # 登录参数
    data = {
        "from": "1",
        "userName": username,
        "password": md5_encrypt(password),
        "platformId": 88,
        "captcha": captcha_text
    }
    # 发送 POST 请求进行登录
    response = requests.post(valid_url, data=data)
    if response.status_code == 200:
        if '验证码校验成功' in response.text:  # 假设页面中有"登录成功"标识
            print("验证码校验成功！")
            return True
        else:
            print("验证码校验失败，验证码可能错误或其他原因。")
            return False
    else:
        print(f"验证码校验请求失败，HTTP状态码: {response.status_code}")
        return False


def login_study(username, password, log_queue):
    if username and password:
        kaptcha_url = f"https://gp.chinahrt.com/gp6/system/manager/kaptcha?d={get_current_time_stamp()}"
        # 获取验证码图片
        captcha_image_path = get_captcha_image(kaptcha_url)
        # 解析验证码
        captcha_text = parse_captcha(captcha_image_path)
        # 校验验证码
        if valid(captcha_text, username, password):
            headers = login(username, password, captcha_text)
            log_queue.put("登录成功，加载课程章节...\n")
            start_study(headers, [my_course_params_bx, my_course_params_xx], log_queue)
        else:
            messagebox.showerror("验证码错误", "验证码错误，请重新输入")
    else:
        messagebox.showerror("输入错误", "请填写用户名和密码")


def create_ui():
    window = tk.Tk()
    window.title("赤峰市继续教育平台在线学习平台")

    # 创建用户名和密码输入框
    tk.Label(window, text="用户名:").grid(row=0, column=0)
    username_entry = tk.Entry(window)
    username_entry.grid(row=0, column=1)
    username_entry.insert(0, "15540039771")

    tk.Label(window, text="密码:").grid(row=1, column=0)
    password_entry = tk.Entry(window, show="*")
    password_entry.grid(row=1, column=1)
    password_entry.insert(0, "Aa123456789")


    # 创建一个日志队列，用于线程间安全地传递消息
    log_queue = queue.Queue()

    # 创建登录按钮
    login_button = tk.Button(window, text="登录学习", command=lambda: on_login_button_click(username_entry, password_entry
                                                                                , login_button, log_queue))
    login_button.grid(row=2, column=0, columnspan=2)
    # 创建一个文本框用于显示学习进度和日志
    result_text = tk.Text(window, width=50, height=20)
    result_text.grid(row=3, column=0, columnspan=2)

    # 启动定时任务，持续更新日志
    update_log(result_text, log_queue)

    # 运行主循环
    window.mainloop()


if __name__ == '__main__':
    create_ui()