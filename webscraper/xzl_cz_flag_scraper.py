import os
import time
import random
import logging
import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from fake_useragent import UserAgent

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建 UserAgent 实例
ua = UserAgent()

# 设置 ChromeOptions
chrome_options = Options()
chrome_options.add_argument(f'--user-agent={ua.random}')
chrome_options.add_argument('--headless')  # 无头模式
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--disable-extensions')
chrome_options.add_argument('--disable-blink-features=AutomationControlled')

# 禁用图片和视频
prefs = {
    "profile.managed_default_content_settings.images": 2,
    "profile.managed_default_content_settings.videos": 2,
    "profile.managed_default_content_settings.stylesheets": 2,
    "profile.managed_default_content_settings.fonts": 2,
    "profile.managed_default_content_settings.cookies": 2
}
chrome_options.add_experimental_option("prefs", prefs)

# 跟踪已完成的分页文件
completed_pages_file = 'completed_pages_xzl_zu.txt'

# 读取已完成的最后一页，以便从中断处恢复
def get_last_completed_page():
    if os.path.exists(completed_pages_file):
        with open(completed_pages_file, 'r') as f:
            last_page = f.read().strip()
            return int(last_page) if last_page.isdigit() else 1
    return 1

# 保存当前完成的页数
def save_completed_page(page):
    with open(completed_pages_file, 'w') as f:
        f.write(str(page))

# 启动 Chrome
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# CSV 文件路径
output_csv = '写字楼租赁11.csv'
fieldnames = ['电话号码', '经纪人姓名', '是否虚拟号码']

# 创建 CSV 文件并写入标题
with open(output_csv, mode='a', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if os.stat(output_csv).st_size == 0:  # 检查文件是否为空
        writer.writeheader()  # 写入标题行

    try:
        # 从记录的最后完成的页数继续
        start_page = get_last_completed_page()
        logging.info(f"从第 {start_page} 页开始爬取。")

        # 循环访问分页
        for page in range(start_page, 527):
            url = f"https://sh.sydc.anjuke.com/xzl-zu/" if page == 1 else f"https://sh.sydc.anjuke.com/xzl-zu/p{page}/?listFilterCount=20&maxInserted=0&skuInserted=0&zzvipInserted=0"
            logging.info(f"访问页面: {url}")
            driver.get(url)
            time.sleep(random.uniform(2, 4))  # 随机等待时间

            # 检查是否有结果提示
            no_result_elem = driver.find_elements(By.CLASS_NAME, 'noresult-tips')
            if no_result_elem:
                logging.info("没有结果，终止分页查询。")
                break

            # 等待列表项加载
            list_items = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, 'list-item'))
            )

            # 收集详情页链接
            detail_links = []
            for item in list_items:
                try:
                    link = item.find_element(By.TAG_NAME, 'a').get_attribute('href')
                    logging.info(f"提取到链接: {link}")
                    detail_links.append(link)
                except Exception as e:
                    logging.error(f"未能提取链接: {e}")

            # 遍历详情页链接并提取信息
            for link in detail_links:
                try:
                    driver.get(link)

                    # 等待电话联系部分加载
                    tel_wrap = WebDriverWait(driver, 3).until(
                        EC.element_to_be_clickable((By.CLASS_NAME, 'tel-wrap'))
                    )

                    # 检查是否包含 'jinpu-detail-bgcolor' 类名
                    if 'jinpu-detail-bgcolor' in tel_wrap.get_attribute('class'):
                        logging.info("电话号码被保护，跳过该条目。")
                        continue  # 跳过该条目

                    tel_wrap.click()  # 点击电话链接

                    # 等待电话信息加载
                    phone_number_elem = WebDriverWait(driver, 3).until(
                        EC.presence_of_element_located((By.CLASS_NAME, 'tel-phone-number'))
                    )
                    phone_number = phone_number_elem.text

                    # 检查是否为虚拟号码
                    is_virtual = '虚拟号' in driver.page_source
                    if is_virtual:
                        logging.info(f"电话号码: {phone_number} (虚拟号码)")
                    else:
                        logging.info(f"电话号码: {phone_number}")

                    # 获取经纪人姓名
                    broker_name = driver.find_element(By.CLASS_NAME, 'name').text
                    logging.info(f"经纪人姓名: {broker_name}")

                    # 写入 CSV 并立即刷新
                    writer.writerow({
                        '电话号码': phone_number,
                        '经纪人姓名': broker_name,
                        '是否虚拟号码': '是' if is_virtual else '否'
                    })
                    csvfile.flush()  # 刷新文件，将数据立即写入磁盘

                except Exception as e:
                    logging.error(f"未能获取电话或经纪人信息: {e}")

            # 记录已完成的分页数
            save_completed_page(page)
            logging.info(f"第 {page} 页完成，记录已更新。")

    except Exception as main_e:
        logging.critical(f"发生错误: {main_e}")
    finally:
        driver.quit()
        logging.info("浏览器已关闭")
