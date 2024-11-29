import base64
import io
from datetime import datetime
import hashlib
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox

import math
import pytesseract
import requests
import vlc
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from PIL import Image, ImageTk

''' 赤峰市专业技术人员继续教育基地'''

START_DATE = datetime(2024, 11, 28)  # 示例日期，你可以选择程序首次运行的日期

# 登录API URL
login_url = "https://manage.yzspeixun.com//yzsApi/platform/login"

def md5_encrypt(password):
    # 使用 MD5 对密码进行加密
    md5 = hashlib.md5()
    md5.update(password.encode('utf-8'))  # 将密码编码为字节流
    return md5.hexdigest()  # 返回加密后的十六进制字符串


def get_captcha_image(headers, kaptcha_url):
    try:
        response = requests.get(kaptcha_url, headers=headers, timeout=10)
        response.raise_for_status()  # 检查请求是否成功

        if response.status_code == 200:
            # 将图片内容转换为PIL对象
            image = Image.open(io.BytesIO(response.content))
            return image
        else:
            messagebox.showerror("错误", f"获取验证码失败，HTTP状态码: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        # 捕获网络请求错误
        messagebox.showerror("错误", f"请求验证码失败: {e}")
        return None


def parse_captcha(captcha_image_path):
    captcha_text = ''
    try:
        # 载入验证码图片
        captcha_image = Image.open(captcha_image_path)

        # 使用 Tesseract 解析验证码
        captcha_text = pytesseract.image_to_string(captcha_image).strip()

        # 输出解析结果
        print(f"解析出的验证码是：{captcha_text}")
    except Exception as e:
        print(f"解析验证码失败: {e}")
    return captcha_text


# 验证验证码
def valid(headers, captcha_text, username, password):
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
    response = requests.post(valid_url, data=data, headers=headers)
    if response.status_code == 200:
        return response.json()


def get_current_time_stamp():
    # 获取当前的日期和时间
    now = datetime.now()
    # 获取时间戳（秒级），并转换为毫秒
    timestamp_milliseconds = int(now.timestamp() * 1000)
    print(f"毫秒级时间戳: {timestamp_milliseconds}")
    return timestamp_milliseconds


def generateHeaders(data):
    headers = {
        "pragma": "no-cache",
        "cache-control": "no-cache",
        "sec-ch-ua-platform": "macOS",
        "cookie": "",
        "hrttoken": "",
        "host": "gp.chinahrt.com",
        # "authorization": data['token'],
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "accept": "application/json, text/plain, */*",
        "sec-ch-ua": "Google Chrome;v=131, Chromium;v=131, Not_A Brand;v=24",
        "content-type": "application/json;charset=UTF-8",
        "sec-ch-ua-mobile": "?0",
        "origin": "https://gp.chinahrt.com",
        "sec-fetch-site": "same-site",
        "sec-fetch-mode": "cors",
        "sec-fetch-dest": "empty",
        "referer": "https://gp.chinahrt.com/index.html",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "zh-CN,zh;q=0.9",
        "priority": "u=1, i"
    }
    return headers


def getSession(headers):
    get_session_url = "https://gp.chinahrt.com/gp6/system/stu/user/getSession"
    session_response = requests.get(get_session_url, headers=headers)
    if session_response.status_code == 200:
        session_data = session_response.json()
        print("获取用户会话信息成功，返回数据:", session_data)
        return session_data
    else:
        print("获取用户会话信息失败，状态码:", session_response.status_code)


def login(username, password, log_queue, captcha):
    global headers
    # 发送请求获取验证码图片
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    if username and password and captcha:
        # 校验验证码
        data = valid(headers, captcha, username, password)
        print(f'登录校验结果：{data}')
        if data and data['success']:
            headers = generateHeaders(data)
            session_data = getSession(headers)
            log_queue.put("登录成功，加载课程章节...\n")
        else:
            messagebox.showerror("登录失败", "验证码或账号密码错误，请重新输入")
    else:
        messagebox.showerror("输入错误", "请填写用户名和密码")
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
    # log_queue.put(f"开始播放视频：{video_url}\n")
    while current_time < video_duration:
        # 假设每秒钟播放1秒，模拟视频播放
        time.sleep(1)
        current_time += 1
        # log_queue.put(f"当前播放进度：{current_time}/{video_duration}秒\n")

    log_queue.put(f"播放视频结束：{video_url}\n")
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
        log_queue.put(f"章节{chapter_id} 完成学习状态更新成功：{video_url}\n")
    else:
        print(f"学习状态更新失败，状态码: {response.status_code}")


def check_trial_period():
    current_date = datetime.today()  # 获取当前日期
    # 计算当前日期与开始日期的差值
    date_diff = (current_date - START_DATE).days

    if date_diff > 30:
        return True  # 超过30天，表示试用期已过
    else:
        return False  # 试用期未过


def on_login_button_click(username_entry, password_entry, login_button, log_queue, captcha):
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

            threading.Thread(target=login_study, args=(username, password, log_queue, captcha), daemon=True).start()  # 启动后台线程
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


def parse_captcha(captcha_image_path):
    # 载入验证码图片
    captcha_image = Image.open(captcha_image_path)

    # 使用 Tesseract 解析验证码
    captcha_text = pytesseract.image_to_string(captcha_image).strip()

    # 输出解析结果
    print(f"解析出的验证码是：{captcha_text}")
    return captcha_text


def get_total_pages(total_size, page_size):
    """
    计算总页数
    """
    return math.ceil(total_size / page_size)


def get_page_data_range(page_index, page_size):
    """
    根据页码计算每页数据的开始和结束索引
    """
    start_index = (page_index - 1) * page_size  # 起始位置
    end_index = start_index + page_size  # 结束位置（不包含）
    return start_index, end_index


def paginate(total_size, page_size, page_index):
    """
    根据总数据量、每页显示的数据量和当前页码进行分页处理，返回当前页数据的范围和总页数。
    """
    # 计算总页数
    total_pages = get_total_pages(total_size, page_size)

    # 判断当前页码是否合法
    if page_index < 1 or page_index > total_pages:
        return None, total_pages  # 当前页码无效，返回空数据

    # 获取当前页的数据范围
    start_index, end_index = get_page_data_range(page_index, page_size)

    # 如果最后一页不足一页数据，则调整结束索引
    if end_index > total_size:
        end_index = total_size

    return (start_index, end_index), total_pages


# 获取课程列表
def get_courses_by_plan_id(headers, plan_id):
    params = {
        "pageSize": 9,
        "curPage": 1,
        "planId": plan_id,  # 替换为实际的 planId
        "learnFinish": 0,
        "status": 2,  # TODO 获取状态为未学习或进行中的课程 1 全部，2 未学习，3 已学习
        "courseTypeId": "",  # 如果没有特定的课程类型，可以为空
    }
    response = requests.get('https://manage.yzspeixun.com//yzsApi/plan/getPlanCourseList', params=params, headers=headers, timeout=10)
    if response.status_code == 200:
        data = response.json()
        if data.get('success') and data.get('data'):
            courses = []
            totalSize = data['data'].get('totalSize', 0)
            pageSize = data['data'].get('pageSize', 0)
            currentPage = data['data'].get('currentPage', 1)
            print(f"获取到{totalSize}个课程")
            # 获取当前页的数据范围和总页数
            data_range, total_pages = paginate(totalSize, pageSize, currentPage)
            for page_index in range(1, total_pages + 1):
                data_range, _ = paginate(totalSize, pageSize, page_index)
                course_items = page_get_courses(headers, page_index, params)
                print(f"第 {page_index} 页，数据范围：{data_range}")
                courses.extend(course_items)
            return courses
        else:
            print(f"响应数据错误：{data.get('msg')}")
            return []
    else:
        print(f"请求失败，状态码：{response.status_code}")
        return []


def page_get_courses(headers, page_index, params):
    params['curPage'] = page_index
    response = requests.get('https://manage.yzspeixun.com//yzsApi/plan/getPlanCourseList', params=params,
                            headers=headers, timeout=10)
    if response.status_code == 200:
        data = response.json()
        if data.get('success') and data.get('data'):
            courses = data['data'].get('list', [])
            return courses
        else:
            print(f"响应数据错误：{data.get('msg')}")
            return []
    else:
        print(f"请求失败，状态码：{response.status_code}")
        return []


# 获取培训计划列表的函数
def get_training_plans(headers):
    url = "https://manage.yzspeixun.com/yzsApi/plan/getPlanList"
    params = {
        "planType": "",  # 可以根据实际情况填充，如果没有可以为空
        "pageSize": 10,
        "curPage": 1,
        "isFinish": 0  # 只获取未完成的计划
    }
    # 发送请求获取培训计划
    response = requests.get(url, params=params, headers=headers, timeout=10)
    if response.status_code == 200:
        data = response.json()
        if data.get('success') and data.get('data'):
            plans = data['data'].get('list', [])
            return plans
        else:
            print(f"响应错误：{data.get('msg')}")
            return []
    else:
        print(f"请求失败，状态码：{response.status_code}")
        return []

# 获取并打印培训计划的ID，供后续查询课程使用
def get_plan_ids(headers):
    plans = get_training_plans(headers)
    plan_ids = [plan['id'] for plan in plans]  # 提取计划ID
    print("获取到的培训计划ID：", plan_ids)
    return plan_ids


def get_video_url(headers, plan_id, course_id, chapter_id, section_id, log_queue):
    url = f'https://manage.yzspeixun.com//yzsApi/course/getVideo?planId={plan_id}&courseId={course_id}&sectionId={section_id}'
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code == 200:
        data = response.json()
        if data.get('success') and data.get('data'):
            videos = data['data']['video']['videoTranscodeLists']
            studyCode = data['data']['studyCode']
            recordId = data['data']['recordId']
            for video in videos:
                log_queue.put(f"获取到的视频链接：{video}\n")
                video_url = video['videoTransUrl']
                videoCode = video['videoCode']
                return studyCode, recordId, videoCode, video_url
        else:
            print(f"响应错误：{data.get('msg')}")
            return None
    else:
        print(f"请求失败，状态码：{response.status_code}")
        return None


def watch_video(headers, section, studyCode, recordId, study_code, video_url, log_queue):

    user_id = ''
    response = requests.post('https://manage.yzspeixun.com//yzsApi/user/userInfo', headers=headers, timeout=10)
    if response.status_code == 200:
        data = response.json()
        if data.get('msg') == 'success':
            user_id = data['data']['userId']
    else:
        log_queue.put(f"获取用户id，状态码：{response.status_code}\n")
    # 模拟视频播放
    def play_video():
        video_duration = section['total_time']  # 假设视频的总时长，单位秒
        current_time = section['study_time']  # 假设当前播放时间
        log_queue.put(f"开始播放视频：{video_url}, 视频开始时间：{current_time}秒\n")
        platformId = 4
        # 当前时间初始化
        current_time = 0
        # 计算每15秒保存进度的时刻
        last_saved_time = 0
        # 模拟视频播放
        while current_time < video_duration:
            # 每秒播放一秒
            time.sleep(1)  # 模拟视频播放的延迟
            current_time += 1

            # 每15秒保存一次进度
            if current_time - last_saved_time >= 15:
                save_view_process_record(platformId, current_time)
                last_saved_time = current_time  # 更新上次保存进度的时间

                # 将播放进度日志输出到日志队列
                log_queue.put(f"当前播放进度：{current_time}/{video_duration}秒\n")

        # 视频播放结束时，最后一次保存进度
        save_view_process_record(platformId, video_duration)
        log_queue.put(f"视频播放结束，最终进度：{current_time}/{video_duration}秒\n")
        # 调用更新进度接口
        # 构造请求参数 section['plan_id'], section['course_id'], section['chapter_id'], section['section_id'],
        # studyCode: this.video.studyCode,
        # isEnd: e || this.video.ended,
        # updateRedisMap: this.video.updateRedisMap,
        # recordId: this.video.recordId,
        # sectionId: this.sectionId,
        # time: this.time,
        # signId: [this.platformId, this.planId, this.userInfo.userId].join("#")
        log_queue.put(f"播放视频结束：{video_url}\n")

    def save_view_process_record(platformId, video_duration):
        data = {
            'studyCode': studyCode,  # 替换为实际值
            'isEnd': True,  # 替换为实际值
            'updateRedisMap': 1,  # 替换为实际值
            'recordId': recordId,  # 替换为实际值
            'sectionId': section['section_id'],  # 替换为实际值
            'time': video_duration,  # 替换为实际值
            'signId': f"{platformId}#{section['plan_id']}#{user_id}"  # 替换为实际值
        }
        print(f'更新进度请求参数:{data}')
        response = requests.post('https://manage.yzspeixun.com//yzsApi/course/takeRecord', params=data, headers=headers,
                                 timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success') and data.get('data') != '违规提交':
                log_queue.put(f"更新进度成功：{video_url}\n")
            else:
                log_queue.put(f"更新进度失败：{data.get('data')}\n")
        else:
            log_queue.put(f"更新进度失败，状态码：{response.status_code}\n")

    play_video()
    # # 创建并启动播放视频的线程
    # play_thread = threading.Thread(target=play_video)
    # play_thread.start()
    # return play_thread  # 返回线程对象，可以在外部管理或等待线程完成


def start_study(headers, log_queue):
    # 获取培训计划
    plan_ids = get_plan_ids(headers)
    print(plan_ids)
    if plan_ids:
        for plan_id in plan_ids:
            print(f"查询计划ID {plan_id} 下的课程：")
            sections = get_sections(headers, plan_id, log_queue)
            if sections:
                for section in sections:
                    print(f"开始学习课程：{section['course_name']}，章节：{section['chapter_name']}")
                    # 获取视频链接
                    studyCode, recordId, videoCode, video_url= get_video_url(headers, section['plan_id'], section['course_id'], section['chapter_id'], section['section_id'], log_queue)
                    if video_url:
                        # 观看视频
                        watch_video(headers, section, studyCode, recordId, videoCode, video_url, log_queue)
    else:
        print("没有可用的培训计划")


def login_study(username, password, log_queue, captcha):
    if username and password:
        headers = login(username, password, log_queue, captcha)
        log_queue.put("登录成功，加载课程章节...\n")
        # start_study(headers, log_queue)
    else:
        messagebox.showerror("输入错误", "请填写用户名和密码")


# 显示验证码图片的窗口
def show_captcha_window(window, login_button, username_entry, password_entry, log_queue):

    global headers
    # 发送请求获取验证码图片
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    kaptcha_url = f"https://gp.chinahrt.com/gp6/system/manager/kaptcha?d={get_current_time_stamp()}"
    captcha_image = get_captcha_image(headers, kaptcha_url)
    if captcha_image:
        # 创建新窗口显示验证码
        captcha_window = tk.Toplevel(window)
        captcha_window.title("请输入验证码")

        # 将PIL图片转换为Tkinter兼容的格式
        captcha_photo = ImageTk.PhotoImage(captcha_image)

        # 创建一个Label来显示验证码图片
        captcha_label = tk.Label(captcha_window, image=captcha_photo)
        captcha_label.image = captcha_photo  # 保存对图像的引用
        captcha_label.pack()

        # 创建输入框让用户输入验证码
        captcha_entry = tk.Entry(captcha_window)
        captcha_entry.pack()

        # 创建一个提交按钮
        def submit_captcha():
            captcha = captcha_entry.get()
            if captcha:
                # 提交验证码并继续登录
                print(f"提交的验证码是：{captcha}")
                captcha_window.destroy()  # 关闭验证码窗口
                on_login_button_click(username_entry, password_entry, login_button, log_queue, captcha)  # 带上验证码继续登录
            else:
                messagebox.showwarning("警告", "请输入验证码。")

        submit_button = tk.Button(captcha_window, text="提交验证码", command=submit_captcha)
        submit_button.pack()

# 创建UI界面
def create_ui():
    window = tk.Tk()
    window.title("赤峰市专业技术人员继续教育基地自动学习平台")

    # 创建用户名和密码输入框
    tk.Label(window, text="用户名:").grid(row=0, column=0)
    username_entry = tk.Entry(window)
    username_entry.grid(row=0, column=1)
    username_entry.insert(0, "150429199706061912")

    tk.Label(window, text="密码:").grid(row=1, column=0)
    password_entry = tk.Entry(window, show="*")
    password_entry.grid(row=1, column=1)
    password_entry.insert(0, "5d6s7w")

    # 创建一个日志队列，用于线程间安全地传递消息
    log_queue = queue.Queue()

    # 创建登录按钮
    login_button = tk.Button(window, text="登录学习", command=lambda: show_captcha_window(window, login_button, username_entry, password_entry, log_queue))
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