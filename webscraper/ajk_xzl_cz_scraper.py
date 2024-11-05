import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from fake_useragent import UserAgent
import random
import logging
import csv

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
# 启动 Chrome
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# 创建 CSV 文件并写入标题
with open('写字楼租赁12.csv', mode='a', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['电话号码', '经纪人姓名', '是否虚拟号码']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    try:
        # 循环访问分页
        for page in range(142, 527):  # 设置较大上限以便停止
            if page == 1:
                url = "https://sh.sydc.anjuke.com/xzl-zu/"
            else:
                url = f"https://sh.sydc.anjuke.com/xzl-zu/p{page}/?listFilterCount=20&maxInserted=0&skuInserted=0&zzvipInserted=0"

            logging.info(f"访问页面: {url}")
            driver.get(url)
            # 等待一段时间，随机化
            time.sleep(random.uniform(2, 4))

            # 检查是否有结果提示
            no_result_elem = driver.find_elements(By.CLASS_NAME, 'noresult-tips')
            if no_result_elem:
                logging.info("没有结果，终止分页查询。")
                break  # 终止分页查询

            # 等待列表项加载
            list_items = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, 'list-item'))
            )
            for item in list_items:
                try:
                    link = item.find_element(By.TAG_NAME, 'a').get_attribute('href')
                    logging.info(f"提取到链接: {link}")

                    # 访问链接
                    driver.get(link)

                    # 等待电话联系部分加载
                    tel_wrap = WebDriverWait(driver, 3).until(
                        EC.element_to_be_clickable((By.CLASS_NAME, 'tel-wrap'))
                    )

                    # 检查是否包含 'jinpu-detail-bgcolor' 类名
                    if 'jinpu-detail-bgcolor' in tel_wrap.get_attribute('class'):
                        logging.info("电话号码被保护，跳过该条目。")
                        continue  # 跳过当前条目并继续处理下一个

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

                    # 返回列表页面
                    driver.back()

                except Exception as e:
                    logging.error(f"未能获取电话或经纪人信息: {e}")
                    driver.back()  # 返回列表页面以便继续处理下一个条目

    except Exception as main_e:
        logging.critical(f"发生错误: {main_e}")
    finally:
        # 关闭浏览器
        driver.quit()
        logging.info("浏览器已关闭")
