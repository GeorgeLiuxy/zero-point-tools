import time

import requests
from bs4 import BeautifulSoup
import json
import re

# 获取页面内容
def get_page_content(url, retries=3, delay=2):
    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # 确保请求成功
            return response.text
        except requests.RequestException as e:
            attempt += 1
            print(f"请求失败: {url}, 错误: {e}")
            if attempt < retries:
                print(f"正在进行重试 ({attempt}/{retries})...")
                time.sleep(delay)
            else:
                print(f"重试次数已达上限，无法获取 {url}")
                return None
    return None

# 从页面 HTML 中提取 JSON 数据
def extract_json_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')

    # 使用正则表达式查找 <script id="__NEXT_DATA__"> 中的 JSON 数据
    script_tag = soup.find('script', {'id': '__NEXT_DATA__', 'type': 'application/json'})
    if script_tag:
        json_data_str = script_tag.string.replace('</script>', '').replace('<script id="__NEXT_DATA__" type="application/json">', '')
        # print(json_data_str)
        # 解析 JSON 数据
        try:
            json_obj = json.loads(json_data_str)
            return json_obj
        except json.JSONDecodeError:
            print("JSON 解析失败")
            return None
    else:
        print("未找到 <script id='__NEXT_DATA__'> 标签")
        return None


# 从 JSON 数据中提取相关参数
def get_result_page_info(json_obj, page_no):
    if json_obj:
        # 假设我们知道需要提取的字段名
        # 例如：获取页面的总结果数
        page_info = json_obj.get('props', {}).get('pageProps', {}).get('page', {}).get(f'search?terms=culture&page={page_no}', None)
        total_page = page_info.get('total', None)
        results = page_info.get('results', {})
        # 你可以在这里获取你需要的其他参数
        print(f"总结果数: {total_page}")
        # 根据你的需要提取其他数据
        return total_page, results
    else:
        print("无效的 JSON 对象")
        return None

# 主程序
def main(url, page_no):
    html = get_page_content(url)  # 获取页面 HTML
    if html:
        json_obj = extract_json_from_html(html)  # 提取 JSON 数据
        total_page, results = get_result_page_info(json_obj, page_no)  # 获取参数值
        return total_page, results
    return None

# 测试
url = 'https://www.bbc.com/search?q=culture'  # 需要爬取的页面 URL
main(url, 0)
