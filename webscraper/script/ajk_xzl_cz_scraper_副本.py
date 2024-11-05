import time
import logging
import csv
import random
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
chrome_options.add_argument(f'--user-agent={ua.random}')  # 随机 User-Agent
chrome_options.add_argument('--headless')  # 无头模式（可根据需要注释掉这一行）
chrome_options.add_argument('--disable-gpu')

# 禁用图片和视频
prefs = {
    "profile.managed_default_content_settings.images": 2,  # 禁用图片加载
    "profile.managed_default_content_settings.videos": 2,   # 禁用视频加载
    "profile.managed_default_content_settings.stylesheets": 2,  # 禁用样式表
    "profile.managed_default_content_settings.fonts": 2,  # 禁用字体加载
    "profile.managed_default_content_settings.cookies": 2  # 禁用Cookies
}
chrome_options.add_experimental_option("prefs", prefs)

# 步骤 1：提取链接并保存到文件
link_file = '../extracted_xzlcz_links.txt'

# 步骤 2：读取文件中的链接并访问，提取电话号码和经纪人信息
with open(link_file, 'r') as f, open('写字楼出租.csv', mode='w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['电话号码', '经纪人姓名', '是否虚拟号码']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    for link in f:
        link = link.strip()  # 去除多余的换行符和空格
        try:
            logging.info(f"访问链接: {link}")
            driver.get(link)

            # 等待电话联系部分加载
            tel_wrap = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.CLASS_NAME, 'tel-wrap'))
            )
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
            csvfile.flush()  # 刷新文件

        except Exception as e:
            logging.error(f"未能获取电话或经纪人信息: {e}")

    # 关闭浏览器
    driver.quit()
    logging.info("浏览器已关闭")
