import asyncio
import datetime
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from selenium.common.exceptions import WebDriverException

START_DATE = datetime.date(2024, 11, 26)  # 示例日期，你可以选择程序首次运行的日期

# 网站和URL的映射关系
website_url_map = {
    "辽宁教育公共服务云平台": "https://peixun.tlsjyy.com.cn/",
    "网站B": "http://example.com/b",
    "网站C": "http://example.com/c"
}

# 本地 chromedriver 路径
chromedriver_path = "./chromedriver"  # 替换为你本地的 chromedriver 路径

user_list = []
# chromedriver_path = "./chromedriver.exe"  # 替换为你本地的 chromedriver 路径
# chrome_binary = "./chrome-win64/chrome.exe"


# 假设你用一个字典来保存当前播放的状态
playback_state = {
    'course_type_bx': False,  # '必修完成页面'
    'course_type_xx': False,  # '选修完成页面'
    'current_course_name': '',  # '课程名称'
    'current_chapter_name': ''  # '章节名称'
}
completed_courses = []

browser_processes = []  # 浏览器进程列表

# 登录验证
def validate_login(username, password):
    return username == "admin" and password == "123456"

# 登录页面函数
def create_login_page():
    # 登录页面主窗口
    root = tk.Tk()
    root.title("登录")
    root.geometry("400x200")  # 初始窗口大小
    root.minsize(300, 150)    # 最小窗口大小

    # 配置行列权重，使得组件能够拉伸
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=2)
    root.rowconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)
    root.rowconfigure(2, weight=1)
    root.rowconfigure(3, weight=1)

    # 标题
    tk.Label(root, text="登录", font=("Arial", 16)).grid(row=0, column=0, columnspan=2, pady=10, sticky="N")

    # 用户名和密码
    tk.Label(root, text="用户名").grid(row=1, column=0, sticky="E", padx=10)
    username_entry = tk.Entry(root)
    username_entry.grid(row=1, column=1, sticky="EW", padx=10)
    username_entry.insert(0, "admin")

    tk.Label(root, text="密码").grid(row=2, column=0, sticky="E", padx=10)
    password_entry = tk.Entry(root, show="*")
    password_entry.grid(row=2, column=1, sticky="EW", padx=10)
    password_entry.insert(0, "123456")

    def check_trial_period():
        current_date = datetime.date.today()  # 获取当前日期
        # 计算当前日期与开始日期的差值
        date_diff = (current_date - START_DATE).days

        if date_diff > 30:
            return True  # 超过30天，表示试用期已过
        else:
            return False  # 试用期未过

    # 按钮
    def login():
        username = username_entry.get()
        password = password_entry.get()
        # 检查试用期是否过期
        if check_trial_period():
            messagebox.showerror("试用期已过期", "您的试用期已过期，程序无法继续使用。请联系管理员")
            sys.exit()  # 程序退出
        if validate_login(username, password):
            root.destroy()
            create_task_page()
        else:
            messagebox.showerror("错误", "用户名或密码错误！")

    tk.Button(root, text="取消", command=root.quit).grid(row=3, column=0, padx=10, pady=10, sticky="E")
    tk.Button(root, text="登录", command=login).grid(row=3, column=1, padx=10, pady=10, sticky="W")

    root.mainloop()


def create_task_page():
    # 创建任务执行窗口
    task_window = tk.Tk()
    task_window.title("自动刷视频系统")
    task_window.geometry("800x600")  # 初始窗口大小
    task_window.minsize(800, 600)    # 最小窗口大小

    # 配置网格权重
    task_window.columnconfigure(0, weight=1)
    task_window.columnconfigure(1, weight=1)
    task_window.columnconfigure(2, weight=1)
    task_window.rowconfigure(1, weight=1)  # 用户列表区域
    task_window.rowconfigure(2, weight=0)  # 按钮
    task_window.rowconfigure(3, weight=2)  # 日志区域

    # 标题
    tk.Label(task_window, text="自动刷视频系统", font=("Arial", 16)).grid(row=0, column=0, columnspan=3, pady=10, sticky="EW")

    # 网站选择
    tk.Label(task_window, text="网站选择").grid(row=1, column=0, sticky="E", padx=10, pady=5)
    # 网站选择与 URL 映射
    tk.Label(task_window, text="网站选择").grid(row=1, column=0, sticky="E", padx=10, pady=5)

    # 网站选择下拉框
    website_combo = ttk.Combobox(task_window, values=list(website_url_map.keys()))
    website_combo.grid(row=1, column=1, columnspan=2, sticky="EW", padx=10, pady=5)

    # URL显示标签
    selected_url = tk.StringVar()  # 用于动态显示 URL
    url_label = tk.Label(task_window, textvariable=selected_url, fg="blue")
    url_label.grid(row=1, column=3, columnspan=2, sticky="W", padx=10, pady=5)

    # 当用户选择网站时，更新 URL
    def update_url(event):
        selected_website = website_combo.get()
        if selected_website in website_url_map:
            selected_url.set(website_url_map[selected_website])  # 显示对应URL
        else:
            selected_url.set("")

    # 绑定下拉框选择事件
    website_combo.bind("<<ComboboxSelected>>", update_url)

    # 用户输入区域
    tk.Label(task_window, text="用户名").grid(row=2, column=0, sticky="E", padx=10, pady=5)
    username_entry = tk.Entry(task_window)
    username_entry.grid(row=2, column=1, sticky="EW", padx=5, pady=5)
    username_entry.insert(0, "15540039771")

    tk.Label(task_window, text="密码").grid(row=2, column=2, sticky="E", padx=10, pady=5)
    password_entry = tk.Entry(task_window)
    password_entry.grid(row=2, column=3, sticky="EW", padx=5, pady=5)
    password_entry.insert(0, "Aa123456789")

    # 用户列表
    tk.Label(task_window, text="用户列表").grid(row=3, column=0, sticky="W", padx=10)
    user_table = ttk.Treeview(task_window, columns=("用户名", "密码"), show="headings")
    user_table.heading("用户名", text="用户名")
    user_table.heading("密码", text="密码")
    user_table.grid(row=3, column=1, columnspan=3, sticky="NSEW", padx=10, pady=5)

    # 添加和删除功能
    def add_user():
        global user_list
        username = username_entry.get().strip()
        password = password_entry.get().strip()

        if not username or not password:
            messagebox.showwarning("警告", "用户名和密码不能为空！")
            return
        # 检查用户是否已存在
        if any(user[0] == username for user in user_list):
            messagebox.showwarning("警告", f"用户名 '{username}' 已存在！")
            return
        # 添加到Treeview和user_list
        user_table.insert("", "end", values=(username, password))
        user_list.append((username, password))  # 更新全局用户列表
        username_entry.delete(0, tk.END)
        password_entry.delete(0, tk.END)
        update_user_list_display()

    # 显示当前用户列表
    def update_user_list_display():
        print("当前用户列表：", user_list)

    # 删除用户功能
    def delete_user():
        global user_list
        selected_rows = user_table.selection()  # 获取选中行的ID
        if not selected_rows:
            messagebox.showwarning("警告", "未选中任何用户！")
            return

        # 删除选中的行，并更新user_list
        for row_id in selected_rows:
            row_data = user_table.item(row_id, "values")
            if row_data in user_list:
                user_list.remove(row_data)  # 从user_list中删除
            user_table.delete(row_id)  # 从Treeview中删除
        update_user_list_display()

    # 异步执行任务的封装函数
    def start_learning_async():
        global user_list
        if not user_list:
            messagebox.showwarning("警告", "用户列表为空，无法开始学习！")
            return
        if not website_combo.get():
            messagebox.showwarning("警告", "未选择待学习网站，无法开始学习！")
            return
        # 启动一个新的线程来执行 start_learning 函数，避免 UI 阻塞
        learning_thread = threading.Thread(target=start_learning, daemon=True)
        learning_thread.start()

    def close_all_browsers():
        for driver in browser_processes:
            try:
                driver.quit()  # 退出浏览器
            except Exception as e:
                print(f"关闭浏览器时出错: {e}")

    # 学习功能
    def start_learning():
        # 模拟学习过程，使用线程并行处理多个账号
        try:
            threads = []
            for username, password in user_list:
                t = threading.Thread(target=process_user, args=(username, password))
                threads.append(t)
                t.start()
            # 等待所有线程完成
            for t in threads:
                t.join()
            time.sleep(1)  # 最后再模拟任务的整体耗时
            write_log("学习任务完成！")
            messagebox.showinfo("提示", "学习任务已完成！")
        except Exception as e:
            # 捕获异常，退出所有浏览器进程，并提示错误信息
            print(f"出现异常: {e}")
            close_all_browsers()
            messagebox.showerror("错误", f"任务执行异常: {str(e)}，所有浏览器已退出，任务重新启动。")
            start_learning_async()  # 重新启动学习任务

    def initialize_browser():
        # 配置 Chrome 浏览器选项
        options = webdriver.ChromeOptions()
        # options.add_argument("--incognito")  # 使用无痕模式
        options.add_argument("--disable-extensions")  # 禁用扩展程序
        options.add_argument("--start-maximized")  # 最大化窗口
        options.add_argument("--disable-extensions")  # 禁用扩展
        options.add_argument("--disable-gpu")  # 禁用 GPU 加速
        options.add_argument("--no-sandbox")  # 不使用沙盒模式
        options.add_argument('--no-proxy-server')  # 禁用代理
        options.add_argument("--disable-blink-features=AutomationControlled")  # 禁用自动化标识
        options.add_argument("disable-infobars")  # 禁用“Chrome正在被自动化测试软件控制”的提示
        options.headless = True  # 或者 False，根据实际需求选择是否使用无头模式
        # options.add_argument("--verbose")  # 启用详细日志
        # options.binary_location = chrome_binary  # 设置 Chrome 浏览器的路径
        # options.add_argument("--window-size=800,600")
        # options.add_argument("--window-position=0,0")  # 将第一个窗口放置在屏幕左上角
        options.add_argument("User-Agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")  # 设置 User-Agent

        # 如果希望显示浏览器窗口，**不要使用** --headless 参数
        # options.add_argument("--headless")  # 注释掉此行，以便显示浏览器窗口
        service = Service(executable_path=chromedriver_path)
        # 启动浏览器，使用本地 chromedriver
        browser = webdriver.Chrome(service=service, options=options)
        return browser

    def process_user(username, password):
        log_message = f"用户 {username} 正在学习..."
        # messagebox.showinfo("提示", "是否已经选择到指定的视频页面，如果没有可能会报错！如果已经完成，点击确认~")
        browser = initialize_browser()
        watch_video(browser, username, password, selected_url.get())
        # 存储浏览器进程以备后续关闭
        browser_processes.append(browser)
        write_log(log_message)
        task_window.update_idletasks()  # 刷新界面
        time.sleep(1)  # 模拟任务耗时

    # 日志功能
    def write_log(message):
        log_text.configure(state="normal")  # 切换为可编辑模式
        log_text.insert("end", f"{message}\n")  # 添加日志
        log_text.configure(state="disabled")  # 切换回只读模式
        log_text.see("end")  # 滚动到最新日志

    tk.Button(task_window, text="添加", command=add_user).grid(row=2, column=4, padx=10, pady=5)
    tk.Button(task_window, text="删除", command=delete_user).grid(row=3, column=4, padx=10, pady=5, sticky="N")

    # 开始学习按钮
    tk.Button(task_window, text="开始批量学习", command=start_learning_async).grid(
        row=4, column=0, columnspan=3, pady=10, sticky="EW"
    )

    # 日志区域
    tk.Label(task_window, text="执行日志").grid(row=5, column=0, sticky="W", padx=10)

    # 滚动条和日志框
    log_frame = tk.Frame(task_window)
    log_frame.grid(row=6, column=0, columnspan=4, sticky="NSEW", padx=10, pady=10)

    # 创建日志区和滚动条
    log_text = tk.Text(log_frame, height=10, wrap="word", state="disabled")
    log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=log_text.yview)

    # 绑定日志框与滚动条
    log_text.configure(yscrollcommand=log_scrollbar.set)
    log_text.pack(side="left", fill="both", expand=True)
    log_scrollbar.pack(side="right", fill="y")

    task_window.mainloop()

# 模拟观看视频的过程
def watch_video(browser, username, password, video_url):

    try:
        # 打开网页等完成登录操作
        login_method(browser, password, username, video_url)
        # 点击并调整任务页
        to_tasks_page(browser)
        # 跳转到课程列表页后，点击“开始学习按钮”，跳转至课程列表页
        # 等待 "开始学习" 按钮加载（根据实际页面情况调整等待时间）
        start_Leaning_btns = get_and_enter_start_learning_btn(browser)
        # 判断是否找到开始学习按钮，如果有则遍历并点击它
        if start_Leaning_btns:
            print(f"找到 {len(start_Leaning_btns)} 个 '开始学习' 按钮")
            # 依次点击每个按钮
            for button in start_Leaning_btns:
                WebDriverWait(browser, 15).until(
                    EC.element_to_be_clickable(button)
                )
                button.click()  # 点击按钮
                time.sleep(2)  # 等待页面跳转（根据实际情况调整）
                # 等待课程列表页加载（假设课程列表页有某个标识元素，可以通过其存在来判断页面加载完成）
                WebDriverWait(browser, 15).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'el-aside'))  # 假设课程列表页有一个元素id为 "course-list"
                )
                click_tab_and_traverse(browser, "必修课程")
                click_tab_and_traverse(browser, "选修课程")
                time.sleep(2)  # 等待返回操作完成
                print("已完成所有课程学习")
        else:
            print("没有找到 '开始学习' 按钮")

    except Exception as e:
        print(f"错误发生在 {username}：", e)
    finally:
        browser.quit()

def click_tab_and_traverse(browser, tab_name):
    # 定位页签并点击
    course_section = WebDriverWait(browser, 15).until(
        EC.element_to_be_clickable((By.CLASS_NAME, 'course_section'))
    )
    # 定位页签并点击
    tab = WebDriverWait(browser, 15).until(
        EC.element_to_be_clickable((By.XPATH, f'//div[@class="tab_box"]//div[contains(text(), "{tab_name}")]'))
    )
    browser.execute_script("arguments[0].scrollIntoView(true);", tab)  # 滚动到目标元素
    ActionChains(browser).move_to_element(tab).click().perform()  # 点击元素
    print(f"已点击 '{tab_name}' 页签")
    # 等待对应页面加载完成
    time.sleep(2)  # 等待页面切换完成，实际情况可以根据需要调整等待时间
    WebDriverWait(browser, 15).until(
        EC.presence_of_element_located((By.CLASS_NAME, 'el-aside'))
    )
    # 获取 el-aside 下的所有 ul 元素
    ul_elements = browser.find_element(By.CLASS_NAME, 'el-aside').find_elements(By.TAG_NAME, 'ul')
    if not ul_elements:
        print(f"没有找到 {tab_name} 页签下的 ul 元素")
        return

    # 遍历每个 ul 元素
    for ul in ul_elements:
        # 获取当前 ul 下的所有 li 元素
        li_elements = ul.find_elements(By.TAG_NAME, 'li')
        print(f"在 {tab_name} 页签下找到 {len(li_elements)} 个 li 元素")
        if not li_elements:
            print(f"该 ul 下没有 li 元素")
            continue

        # 遍历所有 li 元素
        for i, li in enumerate(li_elements):
            # 显式等待，确保第 i 个 li 元素可点击
            li_element = WebDriverWait(browser, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, f"//ul//li[{i + 1}]")
                )
            )
            # 你可以在这里添加你点击后的其他操作（比如等待新页面加载）
            time.sleep(2)  # 示例等待，调整为实际需求
            print(f"在 '{tab_name}' 页签下找到 li 元素:", li_element.text)
            course_type = tab_name  # 课程类型
            course_name = li_element.text  # 课程名称
            browser.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center', inline: 'nearest'});", li_element)  # 滚动到目标元素
            # browser.execute_script("arguments[0].scrollIntoView(true);", li_element)  # 滚动到目标元素
            ActionChains(browser).move_to_element(li_element).click().perform()  # 点击元素
            li_element.click()
            time.sleep(1)  # 等待右侧内容加载
            # 等待右侧内容框加载完成
            WebDriverWait(browser, 15).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'el-main'))
            )
            # 获取右侧框中的章节列表（class="section_ul"）
            section_li = (browser.find_element(By.CLASS_NAME, 'el-main').find_element(By.CLASS_NAME, 'section_ul')
                          .find_elements(By.TAG_NAME, 'li'))
            # 遍历每个章节项
            for j, section in enumerate(section_li):
                # 获取进度条的文本内容
                try:
                    # 获取进度条的文本内容
                    progress_text = WebDriverWait(section, 15).until(
                        EC.presence_of_element_located((By.XPATH, './/div[@class="el-progress__text"]//span'))
                    ).text

                    # 获取章节的标题
                    section_tit = WebDriverWait(section, 15).until(
                        EC.presence_of_element_located((By.CLASS_NAME, 'tit'))
                    )
                    section_text = section_tit.text
                    # 判断进度是否为 100%
                    if progress_text != "100%":
                        # print(f"点击未完成章节: {section_text}，当前进度：{progress_text}")
                        chapter_name = section_text  # 章节名称
                        # time.sleep(3)  # 等待视频页面加载，实际情况可以调整
                        try:
                            # 使用 xpath 定位文本为"1.情境导入"的div元素
                            tit_element = WebDriverWait(browser, 10).until(
                                EC.presence_of_element_located(
                                    (By.XPATH, f"//div[@class='tit' and text()='{section_text}']")
                                )
                            )
                            WebDriverWait(course_section, 10).until(
                                EC.visibility_of(tit_element)  # 等待目标元素变为可见
                            )
                            browser.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center', inline: 'nearest'});", tit_element)
                            WebDriverWait(course_section, 10).until(
                                EC.element_to_be_clickable(tit_element)  # 等待目标元素变为可见
                            )
                            ActionChains(browser).move_to_element(tit_element)  # 点击元素
                            print(f"准备播放:{course_type} - {course_name} - {chapter_name}")
                            # 强制执行 JavaScript 点击
                            browser.execute_script("arguments[0].click();", tit_element)
                            print(f"开始播放:{course_type} - {course_name} - {chapter_name}")
                            time.sleep(3)
                            try:
                                # 等待提示框出现
                                dialog = WebDriverWait(browser, 5).until(
                                    EC.presence_of_element_located((By.CLASS_NAME, 'el-overlay-dialog'))
                                )
                                # 等待确认按钮加载完成
                                confirm_button = dialog.find_element(By.CLASS_NAME, 'el-button--primary')
                                WebDriverWait(browser, 5).until(
                                    EC.element_to_be_clickable(confirm_button)
                                )
                                # 点击确认按钮，跳转视频页面
                                # 滚动到按钮
                                browser.execute_script("arguments[0].scrollIntoView(true);", confirm_button)  # 滚动到目标元素
                                ActionChains(browser).move_to_element(confirm_button).click().perform()  # 点击元素
                                print("关闭提示框，继续跳转到视频页面")
                            except Exception as e2:
                                print(f"未找到提示框，跳过关闭操作: {e2}")
                            print(f"开始学习章节: {section_text}")
                            # 跳转视频页，并处理视频页面播放以及完成逻辑
                            do_view_video(browser, section_text, course_type, course_name, chapter_name)
                        except Exception as e1:
                            print(f"跳转视频页失败{e1}")
                    # else:
                    #     print(f"已完成观看: {course_type} - {course_name} - {section_text}: 已完成学习~~")
                except Exception as e1:
                    print(f"未找到进度信息，跳过该章节: {e1}")
                    continue
            print(f"{course_type} - {course_name}: 已完成学习~~")
            time.sleep(3)  # 适当等待，防止页面过于频繁的请求
        print(f"{tab_name} : 已完成学习~~")
        if course_type == '必修课程':
            playback_state['course_type_bx'] = True
        if course_type == '选修课程':
            playback_state['course_type_xx'] = True


def do_view_video(browser, section_text, course_type, course_name, chapter_name):
    try:
        # 等待提示框出现
        close_dialog(browser)
        # 2. 等待并点击“开始学习”按钮
        # 等待开始学习按钮加载完成
        oper_box = WebDriverWait(browser, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'oper_box'))
        )
        start_button = WebDriverWait(oper_box, 15).until(
            EC.element_to_be_clickable((By.CLASS_NAME, 'button_box'))
        )
        # 点击开始学习按钮
        browser.execute_script("arguments[0].scrollIntoView(true);", start_button)  # 滚动到目标元素
        ActionChains(browser).move_to_element(start_button).click().perform()  # 点击元素
        start_button.click()
        print(f"开始播放视频~~{section_text}")
        # 1. 检查视频是否播放完成
        # 获取视频播放器元素，假设它是一个 <video> 标签
        video_element = WebDriverWait(browser, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, 'video'))
        )
        # 轮询检查视频是否播放完成
        while True:
            # 判断视频是否播放完成（视频播放结束时，ended属性为True）
            is_ended = browser.execute_script('return arguments[0].ended;', video_element)
            if is_ended:
                print("视频播放完毕")
                break  # 视频播放完成，跳出循环
            time.sleep(1)  # 每秒检查一次
        # 等待“结束学习”按钮可点击
        end_button = WebDriverWait(browser, 15).until(
            EC.element_to_be_clickable((By.CLASS_NAME, 'button_box2'))
        )
        # 点击“结束学习”按钮
        end_button.click()
        print(f"{section_text} ~~ 视频播放完成，点击结束学习按钮")
        # 记录已完成的章节
        completed_courses.append((course_type, course_name, chapter_name))
        # 在视频结束后调用 back() 返回上一页面
        browser.back()  # 返回上一页面
        # 判断当前进度
        if playback_state['course_type_bx'] == False:  # 必修课程 没结束，则继续
            click_tab_and_traverse(browser, '必修课程')
        else:
            click_tab_and_traverse(browser, '选修课程')
    except Exception as e:
        print(f"视频播放或结束学习失败: {e}")


def close_dialog(browser):
    try:
        WebDriverWait(browser, 3).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'el-overlay-message-box'))
        )
        # 获取并点击关闭按钮
        close_button = WebDriverWait(browser, 3).until(
            EC.element_to_be_clickable((By.CLASS_NAME, 'el-button--primary'))
        )
        close_button.click()
    except Exception as e1:
        print(f"无提示框，跳过关闭操作:{e1}")


def to_tasks_page(browser):
    # 定义图片的 CSS 选择器或 Xpath
    image_xpath = "//img[@src='https://peixunapi.tlsjyy.com.cn/uploads/images/20241105/9cfd54629a431444812de435bb642fe3.png']"
    # 获取当前窗口的句柄
    current_window = browser.current_window_handle
    try:
        # 登录成功后出框出现弹出框并点击关闭按钮
        WebDriverWait(browser, 15).until(
            EC.visibility_of_element_located((By.CLASS_NAME, 'el-button--primary'))  # 修改为实际的弹出框关闭按钮的定位方式
        )
        close_button = browser.find_element(By.CLASS_NAME, 'el-button--primary')  # 关闭按钮的类名
        WebDriverWait(browser, 15).until(
            EC.element_to_be_clickable(close_button)
        )
        close_button.click()  # 关闭弹出框
        time.sleep(1)  # 等待弹出框关闭
        # 等待图片元素加载并可见
        image_element = WebDriverWait(browser, 15).until(
            EC.visibility_of_element_located((By.XPATH, image_xpath))
        )

        # 检查图片是否加载完成
        if image_element.get_attribute("naturalWidth") != "0":
            WebDriverWait(browser, 15).until(
                EC.element_to_be_clickable(image_element)
            )
            print("图片加载完成，开始点击")
            image_element.click()  # 点击图片
            # 切换到新打开的标签页
            WebDriverWait(browser, 15).until(EC.new_window_is_opened)  # 等待新窗口/标签页打开
            new_window = [window for window in browser.window_handles if window != current_window][0]  # 获取新标签页的句柄
            browser.switch_to.window(new_window)  # 切换到新标签页
        else:
            print("图片未加载完成")
    except Exception as e:
        print(f"发生错误: {e}")
        browser.quit()

def login_method(browser, password, username, video_url):
    try:
        browser.get(video_url)
        # 等待弹出框出现并关闭弹出框
        WebDriverWait(browser, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'el-overlay-dialog'))
        )
        close_button = WebDriverWait(browser, 15).until(
            EC.element_to_be_clickable((By.CLASS_NAME, 'el-button--primary'))
        )
        close_button.click()  # 关闭弹出框
        print("关闭登录页提示弹出框")
        # 等待登录按钮可点击
        login_button = WebDriverWait(browser, 15).until(
            EC.element_to_be_clickable((By.CLASS_NAME, 'login_btn'))
        )
        login_button.click()  # 点击登录按钮
        print("点击登录按钮")
        # 等待用户名输入框并输入用户名
        username_field = WebDriverWait(browser, 15).until(
            EC.element_to_be_clickable((By.XPATH, '//input[@placeholder="请输入手机号"]'))
        )
        username_field.send_keys(username)
        # 等待密码输入框并输入密码
        password_field = WebDriverWait(browser, 15).until(
            EC.element_to_be_clickable((By.XPATH, '//input[@placeholder="请输入密码"]'))
        )
        password_field.send_keys(password)
        password_field.send_keys(Keys.RETURN)  # 提交登录表单
        print("提交登录表单")
        # 等待页面加载完成
        WebDriverWait(browser, 15).until(
            EC.url_changes(video_url)  # 等待页面 URL 发生变化，意味着页面加载完成
        )
    except Exception as e:
        print(f"登录过程发生错误: {e}")
        # 可能需要增加一些重试机制
        # 例如使用递归重试或通知用户

def get_and_enter_start_learning_btn(browser):
    WebDriverWait(browser, 15).until(
        EC.visibility_of_element_located((By.CLASS_NAME, 'taskList_box'))  # 修改为实际的弹出框关闭按钮的定位方式
    )
    buttons = []
    try:
        WebDriverWait(browser, 15).until(
            EC.visibility_of_element_located((By.CLASS_NAME, 'custom-icon-study'))  # 修改为实际的弹出框关闭按钮的定位方式
        )
        # 查找 main_box taskList_box 容器
        buttons = browser.find_elements(By.CLASS_NAME, 'custom-icon-study')
    except Exception as e:
        print("未找到 custom-icon-study 或按钮:", e)
        browser.quit()
    # 打印按钮数量或进行其他操作
    print(f"找到 {len(buttons)} 个按钮")
    return buttons


# 示例：后续操作函数
def proceed_with_url(selected_url):
    current_url = selected_url.get()
    if not current_url:
        messagebox.showwarning("警告", "请选择一个有效的网站！")
    else:
        messagebox.showinfo("提示", f"即将访问：{current_url}")
        # 在这里添加后续处理逻辑，比如打开 URL、发起请求等

def restart_program():
    """ 当程序异常中断时，自动重启程序 """
    messagebox.showinfo("程序重启", "程序即将重启！")
    time.sleep(1)
    subprocess.Popen([sys.executable, 'main.py'])  # 使用当前 Python 解释器启动 main.py


if __name__ == '__main__':
    try:
        # 运行程序
        create_login_page()
    except Exception as e:
        # 捕获并处理其他异常
        print(f"主程序发生异常: {e}")
        restart_program()  # 发生异常时重启程序