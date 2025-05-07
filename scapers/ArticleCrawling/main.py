import requests
from bs4 import BeautifulSoup
import os

import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# 获取网页内容
def get_page_content(url, retries=3, delay=2):
    """
    获取网页内容，加入重试机制。

    :param url: 需要请求的 URL 地址
    :param retries: 最大重试次数
    :param delay: 每次重试之间的等待时间（秒）
    :return: 网页内容，如果请求失败则返回 None
    """
    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url)
            response.raise_for_status()  # 确保请求成功
            return response.text
        except requests.RequestException as e:
            attempt += 1
            print(f"请求失败: {url}, 错误: {e}")
            if attempt < retries:
                print(f"正在进行重试 ({attempt}/{retries})...")
                time.sleep(delay)  # 等待一定时间后重试
            else:
                print(f"重试次数已达上限，无法获取 {url}")
                return None
    return None  # 如果重试次数用尽，仍然返回 None


# 解析网页并提取文章链接
def extract_article_links(html):
    soup = BeautifulSoup(html, 'html.parser')
    content = soup.find('div', {'data-testid': 'new-jersey-grid'})  # 搜索主列表块

    if content:
        links = []
        for a_tag in content.find_all('a', href=True):
            href = a_tag['href']
            if href.startswith('/news/articles'):
                links.append(f'https://www.bbc.com{href}')  # 拼接完整链接
        return links
    else:
        print("未找到文章列表")
        return []


# 解析单个文章页面并提取正文内容
def extract_article_content(html):
    soup = BeautifulSoup(html, 'html.parser')
    # 提取标题
    title_tag = soup.find('h1')
    title = title_tag.get_text(strip=True) if title_tag else "无标题"
    content = soup.find('main', {'id': 'main-content'})

    if content:
        # 获取所有段落并连接
        paragraphs = content.find_all('p')
        text = '\n\n'.join([para.get_text(strip=True) for para in paragraphs])
        return title, text
    else:
        logging.info("未找到正文内容")
        return None, None


# 保存内容到 TXT 文件
def save_to_txt(title, content, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(f"标题: {title}\n\n")
        file.write(content)
    logging.info(f"TXT 文件已保存: {filename}")


# 递归处理文章
def process_articles_links(url):
    # 获取页面内容
    html = get_page_content(url)
    if not html:
        return
    # 提取页面中的文章链接
    article_links = extract_article_links(html)
    if not article_links:
        logging.info("未找到任何文章链接")
        return
    # 遍历文章链接并保存内容
    all_article_links = []
    for link in article_links:
        get_inner_articles_links(link, all_article_links)
    article_links.extend(all_article_links)
    return article_links


def process_articles(links):
    if not os.path.exists('articles'):
        os.makedirs('articles')
    for link in links:
        # logging.info(f"正在处理: {link}")
        article_html = get_page_content(link)
        if article_html:
            title, content = extract_article_content(article_html)
            if content:
                word_count = len(content.split())
                safe_title = title.replace('/', '-').replace(':', '-').replace(' ', '_')
                filename = f"articles/{safe_title}_wordcount_{word_count}.txt"
                save_to_txt(title, content, filename)


def get_inner_articles_links(link, links=[]):
    # 提取页面中的文章链接
    if link:
        # logging.info(f"正在处理: {link}")
        article_html = get_page_content(link)
        # 获取并处理 "Related" 部分中的文章链接
        related_links = extract_related_article_links(article_html)
        # logging.info(f"正在处理相关文章: {related_links}")
        links.extend(related_links)
        # 获取并处理 "More" 部分中的文章链接
        more_links = extract_more_article_links(article_html)
        # logging.info(f"正在处理 'More' 部分的文章: {more_links}")
        links.extend(more_links)


# 获取 "More" 部分中的文章链接
def extract_more_article_links(html):
    soup = BeautifulSoup(html, 'html.parser')
    more_section = soup.find('aside', {'data-analytics_group_name': 'More'})

    if more_section:
        # 获取 <div data-testid="alaska-grid"> 中的所有文章链接
        links = []
        alaska_grid = more_section.find('div', {'data-testid': 'alaska-grid'})
        if alaska_grid:
            for a_tag in alaska_grid.find_all('a', href=True):
                href = a_tag['href']
                if href.startswith('/news/articles'):
                    links.append(f'https://www.bbc.com{href}')  # 拼接完整链接
        return links
    else:
        logging.info("未找到 'More' 部分")
        return []


# 获取 "Related" 部分中的文章链接
def extract_related_article_links(html):
    soup = BeautifulSoup(html, 'html.parser')
    related_section = soup.find('aside', {'data-analytics_group_name': 'Related'})
    if related_section:
        # 获取 <div data-testid="ohio-grid-3"> 中的所有文章链接
        links = []
        ohio_grid = related_section.find('div', {'data-testid': 'ohio-grid-3'})
        if ohio_grid:
            for a_tag in ohio_grid.find_all('a', href=True):
                href = a_tag['href']
                if href.startswith('/news/articles'):
                    links.append(f'https://www.bbc.com{href}')  # 拼接完整链接
        return links
    else:
        logging.info("未找到相关文章部分")
        return []


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
            response = requests.get(url)
            # 判断响应状态码
            if response.status_code == 200:
                logging.info(f"URL 访问成功: {url}")
                return True
            else:
                logging.info(f"URL 访问失败，状态码: {response.status_code}")
                return False
        except requests.RequestException as e:
            attempt += 1
            logging.info(f"请求失败: {url}, 错误: {e}")
            if attempt < retries:
                logging.info(f"正在进行重试 ({attempt}/{retries})...")
                time.sleep(delay)  # 等待一定时间后重试
            else:
                logging.info(f"重试次数已达上限，无法访问 {url}")
                return False
    return False  # 如果重试次数用尽，仍然返回 False


# 主程序
def main(base_url):
    os.makedirs('articles', exist_ok=True)  # 确保保存文件的文件夹存在
    # 开始处理文章
    links = process_articles_links(base_url)
    logging.info(len(links))
    result = list(set(links))
    logging.info(len(result))
    # logging.info(result)
    process_articles(result)


if __name__ == "__main__":
    search_term = 'social'
    for i in range(0, 1500):
        base_url = f'https://www.bbc.com/search?q={search_term}&page={i}'  # 搜索页面的URL
        if check_url_status(base_url):
            main(base_url)
        else:
            logging.info("URL 访问失败，循环结束")
            break
