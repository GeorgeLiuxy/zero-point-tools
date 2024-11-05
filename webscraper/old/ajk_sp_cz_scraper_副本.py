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

# 启动 Chrome
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# 步骤 1：提取链接并保存到文件
link_file = 'extracted_links2.txt'

with open(link_file, 'w') as f:
    try:
        for page in range(298, 527):  # 设置较大上限以便停止
            if page == 1:
                url = "https://sh.sydc.anjuke.com/sp-zu/?from=navigation"
            else:
                url = f"https://sh.sydc.anjuke.com/sp-zu/p{page}/?listFilterCount=5&maxInserted=10&skuInserted=0&zzvipInserted=0"

            logging.info(f"访问页面: {url}")
            driver.get(url)
            time.sleep(random.uniform(1, 3))  # 随机等待时间

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
                    f.write(link + '\n')  # 将链接写入文件
                    logging.info(f"提取到链接: {link}")
                except Exception as e:
                    logging.error(f"未能提取链接: {e}")

    except Exception as main_e:
        logging.critical(f"发生错误: {main_e}")
    finally:
        logging.info("链接提取完成，浏览器已关闭")
        driver.quit()
#
# # 步骤 2：读取文件中的链接并访问，提取电话号码和经纪人信息
# with open(link_file, 'r') as f, open('output_spcz2.csv', mode='w', newline='', encoding='utf-8') as csvfile:
#     fieldnames = ['电话号码', '经纪人姓名', '是否虚拟号码']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#
#     driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
#
#     for link in f:
#         link = link.strip()  # 去除多余的换行符和空格
#         try:
#             logging.info(f"访问链接: {link}")
#             driver.get(link)
#
#             # 等待电话联系部分加载
#             tel_wrap = WebDriverWait(driver, 3).until(
#                 EC.element_to_be_clickable((By.CLASS_NAME, 'tel-wrap'))
#             )
#             tel_wrap.click()  # 点击电话链接
#
#             # 等待电话信息加载
#             phone_number_elem = WebDriverWait(driver, 3).until(
#                 EC.presence_of_element_located((By.CLASS_NAME, 'tel-phone-number'))
#             )
#             phone_number = phone_number_elem.text
#
#             # 检查是否为虚拟号码
#             is_virtual = '虚拟号' in driver.page_source
#             if is_virtual:
#                 logging.info(f"电话号码: {phone_number} (虚拟号码)")
#             else:
#                 logging.info(f"电话号码: {phone_number}")
#
#             # 获取经纪人姓名
#             broker_name = driver.find_element(By.CLASS_NAME, 'name').text
#             logging.info(f"经纪人姓名: {broker_name}")
#
#             # 写入 CSV 并立即刷新
#             writer.writerow({
#                 '电话号码': phone_number,
#                 '经纪人姓名': broker_name,
#                 '是否虚拟号码': '是' if is_virtual else '否'
#             })
#             csvfile.flush()  # 刷新文件
#
#         except Exception as e:
#             logging.error(f"未能获取电话或经纪人信息: {e}")

    # 关闭浏览器
    driver.quit()
    logging.info("浏览器已关闭")
