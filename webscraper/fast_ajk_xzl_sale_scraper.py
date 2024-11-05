import time
import concurrent.futures
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

# 创建 CSV 文件并写入标题
with open('output_xzlcz.csv', mode='w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['电话号码', '经纪人姓名', '是否虚拟号码']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    def fetch_details(link):
        """获取电话号码和经纪人姓名的函数"""
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        try:
            driver.get(link)
            # 等待电话联系部分加载
            tel_wrap = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CLASS_NAME, 'tel-wrap'))
            )
            tel_wrap.click()  # 点击电话链接

            # 等待电话信息加载
            phone_number_elem = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'tel-phone-number'))
            )
            phone_number = phone_number_elem.text

            # 检查是否为虚拟号码
            is_virtual = '虚拟号' in driver.page_source
            logging.info(f"提取到电话号码: {phone_number} {'(虚拟号码)' if is_virtual else ''}")

            # 获取经纪人姓名
            broker_name = driver.find_element(By.CLASS_NAME, 'name').text
            logging.info(f"经纪人姓名: {broker_name}")

            return {
                '电话号码': phone_number,
                '经纪人姓名': broker_name,
                '是否虚拟号码': '是' if is_virtual else '否'
            }

        except Exception as e:
            logging.error(f"未能获取电话或经纪人信息: {e}")
            return None
        finally:
            driver.quit()

    try:
        # 循环访问分页
        for page in range(1, 1000):  # 设置较大上限以便停止
            if page == 1:
                url = "https://sh.sydc.anjuke.com/xzl-shou/?from=navigation"
            else:
                url = (f"https://sh.sydc.anjuke.com/xzl-shou/p{page}/?listFilterCount=24&maxInserted=0&skuInserted=0&zzvipInserted=0")

            logging.info(f"访问页面: {url}")
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            driver.get(url)

            # 检查是否有结果提示
            no_result_elem = driver.find_elements(By.CLASS_NAME, 'noresult-tips')
            if no_result_elem:
                logging.info("没有结果，终止分页查询。")
                break  # 终止分页查询

            # 等待列表项加载
            list_items = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, 'list-item'))
            )
            hrefs = []
            for item in list_items:
                try:
                    link = item.find_element(By.TAG_NAME, 'a').get_attribute('href')
                    hrefs.append(link)
                    logging.info(f"提取到链接: {link}")
                except Exception as e:
                    logging.error(f"未能提取链接: {e}")

            # 使用多线程处理每个链接
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                results = list(executor.map(fetch_details, hrefs))

            # 写入 CSV
            for result in results:
                if result:  # 确保结果不为 None
                    writer.writerow(result)

            driver.quit()

    except Exception as main_e:
        logging.critical(f"发生错误: {main_e}")
    finally:
        logging.info("爬取完成，浏览器已关闭")
