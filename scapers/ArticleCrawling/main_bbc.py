import requests
from bs4 import BeautifulSoup
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.parse

import requests
from bs4 import BeautifulSoup
import json
import re
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# 获取网页内容
def get_page_content(url, retries=3, delay=2):
    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # 确保请求成功
            if response.status_code == 500:
                logging.warning(f"服务器错误 (500), 正在重试: {url}")
                attempt += 1
                time.sleep(delay * attempt)  # 增加等待时间，防止请求过快
                continue
            return response.text  # 返回正常的网页内容
        except requests.RequestException as e:
            attempt += 1
            logging.warning(f"请求失败: {url}, 错误: {e}")
            if attempt < retries:
                logging.info(f"正在进行重试 ({attempt}/{retries})...")
                time.sleep(delay * attempt)  # 每次重试增加等待时间
            else:
                logging.error(f"重试次数已达上限，无法获取 {url}")
                return None
    return None  # 如果重试次数用尽，仍然返回 None


# 解析网页并提取文章链接
def extract_article_links(html):
    soup = BeautifulSoup(html, 'html.parser')
    content = soup.find('div', {'data-testid': 'new-jersey-grid'})  # 搜索主列表块
    if content:
        links = [f'https://www.bbc.com{a_tag["href"]}' for a_tag in content.find_all('a', href=True) if a_tag['href'].startswith('/news/articles')]
        return links
    logging.warning("未找到文章列表")
    return []


# 解析单个文章页面并提取正文内容
def extract_article_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    title_tag = soup.find('h1')
    title = title_tag.get_text(strip=True) if title_tag else "无标题"
    content = soup.find('main', {'id': 'main-content'})
    if content:
        paragraphs = content.find_all('p')
        text = '\n\n'.join([para.get_text(strip=True) for para in paragraphs])
        return title, text
    logging.warning("未找到正文内容")
    return None, None


# 保存内容到 TXT 文件
def save_to_txt(title, content, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(f"标题: {title}\n\n{content}")
    logging.info(f"TXT 文件已保存: {filename}")


# 获取 "More" 部分中的文章链接
def extract_more_article_links(html):
    soup = BeautifulSoup(html, 'html.parser')
    more_section = soup.find('aside', {'data-analytics_group_name': 'More'})
    links = []
    if more_section:
        alaska_grid = more_section.find('div', {'data-testid': 'alaska-grid'})
        if alaska_grid:
            links = [f'https://www.bbc.com{a_tag["href"]}' for a_tag in alaska_grid.find_all('a', href=True) if a_tag['href'].startswith('/news/articles')]
    if not links:
        logging.warning("未找到 'More' 部分")
    return links


# 获取 "Related" 部分中的文章链接
def extract_related_article_links(article_url):
    if article_url:
        article_html = get_page_content(article_url)
        # 使用 rfind() 查找最后一个 '/' 的位置
        last_slash_index = article_url.rfind('/')
        # 截取最后一个 '/' 后面的部分
        article_id = article_url[last_slash_index + 1:]
        if article_html:
            soup = BeautifulSoup(article_html, 'html.parser')
            # 使用正则表达式查找 <script id="__NEXT_DATA__"> 中的 JSON 数据
            script_tag = soup.find('script', {'id': '__NEXT_DATA__', 'type': 'application/json'})
            if script_tag:
                json_data_str = script_tag.string.replace('</script>', '').replace('<script id="__NEXT_DATA__" type="application/json">', '')
                # 解析 JSON 数据
                try:
                    json_obj = json.loads(json_data_str)
                    # 直接访问需要的字段，避免多次查找字典键
                    article_data = json_obj.get('props', {}).get('pageProps', {}).get('page', {})
                    # 检查 'relatedArticles' 是否存在，并提取相关内容
                    related_articles_key = f'@"news","articles","{article_id}",'
                    if related_articles_key in article_data:
                        print(article_data.get(related_articles_key, {}).get('contents', {}))
                        related_articles = article_data.get(related_articles_key, {}).get('contents', {}).get('onwardJourney', {}).get('content', None)
                        # 如果存在相关文章，生成链接
                        if related_articles:
                            links = [f'https://www.bbc.com{article["href"]}' for article in related_articles if 'href' in article]
                            return links
                        else:
                            logging.warning("未找到相关文章内容")
                    else:
                        logging.warning(f"未找到 '{related_articles_key}' 键")
                    return []
                except json.JSONDecodeError:
                    print("JSON 解析失败")
                    return None
            else:
                print("未找到 <script id='__NEXT_DATA__'> 标签")
                return None

def check_url_status(url, retries=3, delay=2):
    """
    检查 URL 是否可访问，加入重试机制。
    :param url: 需要检查的 URL 地址
    :param retries: 最大重试次数
    :param delay: 每次重试之间的等待时间（秒）
    :return: 如果 URL 访问成功返回 True，否则返回 False
    """
    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                logging.info(f"URL 访问成功: {url}")
                return True
            else:
                logging.warning(f"URL 访问失败，状态码: {response.status_code}")
                return False
        except requests.RequestException as e:
            attempt += 1
            logging.warning(f"请求失败: {url}, 错误: {e}")
            if attempt < retries:
                logging.info(f"正在进行重试 ({attempt}/{retries})...")
                time.sleep(delay)
            else:
                logging.error(f"重试次数已达上限，无法访问 {url}")
                return False
    return False


# 处理文章链接
def do_link(base_url, folder, search_term, page_no):
    os.makedirs(folder, exist_ok=True)  # 确保保存文件的文件夹存在
    links = process_search_articles_links(base_url, search_term, page_no)
    if links:
        result = list(set(links))  # 去重
        logging.info(f"处理 {len(result)} 个链接")
        process_articles(result, folder)


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
def get_result_page_info(json_obj, search_term, page_no):
    if json_obj:
        encoded_search_term = urllib.parse.quote_plus(search_term)  # 使用 quote_plus 来编码空格为 "+"，这是 URL 查询参数中常见的格式
        page_info = json_obj.get('props', {}).get('pageProps', {}).get('page', {}).get(f'search?terms={encoded_search_term}&page={page_no}', None)
        total_page = page_info.get('total', None)
        results = page_info.get('results', {})
        # 你可以在这里获取你需要的其他参数
        print(f"总结果数: {total_page}")
        # 根据你的需要提取其他数据
        return total_page, results
    else:
        print("无效的 JSON 对象")
        return None


def process_search_articles_links(url, search_term, page_no):
    result_json = {}
    # 获取页面内容
    html = get_page_content(url)  # 获取搜索结果页面 HTML
    if html:
        json_obj = extract_json_from_html(html)  # 提取 JSON 数据
        total_page, results = get_result_page_info(json_obj, search_term, page_no)  # 获取参数值

        for result in results:
            related_articles = extract_related_article_links(f'https://www.bbc.com{result.get("href")}')
            result_json = {
                'title': result.get('title'),
                'url': result.get('href'),
                'topic': result.get('metadata', {}).get('topics', ''),
                'related_articles': related_articles,
            }
    return result_json


# 处理文章
def process_articles(links, folder):
    for link in links:
        article_html = get_page_content(link)
        if article_html:
            title, content = extract_article_content(article_html)
            if content:
                word_count = len(content.split())
                safe_title = title.replace('/', '-').replace(':', '-').replace(' ', '_')
                filename = f"{folder}/{safe_title}_wordcount_{word_count}.txt"
                save_to_txt(title, content, filename)


# 检查 URL 是否有效，检查搜索结果的最大页数
def get_max_pages(search_term, page_no=0):
    try:
        encoded_search_term = urllib.parse.quote_plus(search_term)
        base_url = f'https://www.bbc.com/search?q={encoded_search_term}'
        html = get_page_content(base_url)
        if html:
            json_obj = extract_json_from_html(html)  # 提取 JSON 数据
            total_page, results = get_result_page_info(json_obj, search_term, page_no)  # 获取参数值
            return total_page
        else:
            logging.error(f"无法获取搜索结果页面: {base_url}")
            return 0
    except AttributeError:
        logging.warning("未找到分页信息，假定只有一页")
        return 1


# 主程序
def main(search_term):
    max_pages = get_max_pages(search_term, 0)  # 获取最大页数
    if max_pages == 0:
        logging.error(f"无法获取 {search_term} 的最大页数")
        return

    for i in range(0, max_pages):  # 根据最大页数动态调整
        encoded_search_term = urllib.parse.quote_plus(search_term)  # 使用 quote_plus 来编码空格为 "+"，这是 URL 查询参数中常见的格式
        logging.info(f"正在处理第 {i} 页")
        if i > 0:
            base_url = f'https://www.bbc.com/search?q={encoded_search_term}&page={i}'  # 搜索页面的URL
        else:
            base_url = f'https://www.bbc.com/search?q={encoded_search_term}'
        # 检查页面是否有效
        if check_url_status(base_url):
            folder = f'result/{search_term}'  # 为每个关键词创建一个独立文件夹
            do_link(base_url, folder, search_term, i)
        else:
            logging.info(f"URL 访问失败，停止处理关键词: {search_term}")
            break


if __name__ == "__main__":
    search_terms = [
        'logic system', 'social etiquette', 'daily life', 'social customs', 'cultural norms',
        'personal behavior', 'legal rights', 'family traditions', 'work-life balance', 'human rights issues',
        'civic responsibilities', 'social justice', 'education system', 'public behavior', 'gender roles',
        'community service', 'international law', 'conflict resolution', 'child-rearing practices', 'marriage laws',

        # 相关社会领域扩展
        'family values', 'ethical behavior', 'civil rights', 'public morals', 'workplace ethics',
        'gender equality', 'cultural diversity', 'social interaction', 'cultural sensitivity', 'youth behavior',
        'interpersonal communication', 'public health', 'privacy rights', 'legal systems', 'rights protection',
        'law enforcement', 'government policy', 'societal issues', 'global citizenship', 'climate justice',

        # 扩展到具体习惯、权利等
        'digital rights', 'freedom of speech', 'political participation', 'religious freedom', 'cultural appropriation',
        'human trafficking', 'immigration laws', 'labor laws', 'equal opportunity', 'consumer rights',
        'data protection', 'cyber security', 'mental health awareness', 'age discrimination', 'social media impact',
        'public policy', 'social equity',
        'income inequality', 'poverty alleviation', 'environmental justice',

        # 更多文化和社会层面
        'intergenerational relationships', 'elderly care',
        'urbanization', 'sustainable living', 'intercultural communication',
        'political correctness', 'sustainable development', 'global warming', 'racial equality', 'diversity inclusion',
        'peacekeeping', 'religion and politics', 'disability rights', 'workplace diversity', 'social mobility',
        'urban poverty', 'education reform', 'gender-based violence',
        'youth empowerment', 'mental health stigma',
        'voting rights', 'charity and philanthropy',
        'economic development', 'housing rights', 'discrimination',
        'international relations', 'social integration', 'collective bargaining', 'affordable healthcare', 'public welfare'
    ]
    for search_term in search_terms:
        main(search_term)
    # with ThreadPoolExecutor(max_workers=1) as executor:  # 限制最大并发线程数为 5
    #     executor.map(main, search_terms)
