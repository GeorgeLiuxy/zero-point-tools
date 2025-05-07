import requests
from bs4 import BeautifulSoup
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.parse
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
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # 确保请求成功
            return response.text
        except requests.RequestException as e:
            attempt += 1
            logging.warning(f"请求失败: {url}, 错误: {e}")
            if attempt < retries:
                logging.info(f"正在进行重试 ({attempt}/{retries})...")
                time.sleep(delay)  # 等待一定时间后重试
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


# 获取更多和相关文章链接
def get_inner_articles_links(link, links=[]):
    if link:
        article_html = get_page_content(link)
        if article_html:
            related_links = extract_related_article_links(article_html)
            links.extend(related_links)
            more_links = extract_more_article_links(article_html)
            links.extend(more_links)


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
def extract_related_article_links(html):
    soup = BeautifulSoup(html, 'html.parser')
    related_section = soup.find('aside', {'data-analytics_group_name': 'Related'})
    links = []
    if related_section:
        ohio_grid = related_section.find('div', {'data-testid': 'ohio-grid-3'})
        if ohio_grid:
            links = [f'https://www.bbc.com{a_tag["href"]}' for a_tag in ohio_grid.find_all('a', href=True) if a_tag['href'].startswith('/news/articles')]
    if not links:
        logging.warning("未找到相关文章部分")
    return links


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
def do_link(base_url, folder):
    os.makedirs(folder, exist_ok=True)  # 确保保存文件的文件夹存在
    links = process_articles_links(base_url)
    if links:
        result = list(set(links))  # 去重
        logging.info(f"处理 {len(result)} 个链接")
        process_articles(result, folder)


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


# 主程序
def main(search_term):
    for i in range(0, 1500):
        encoded_search_term = urllib.parse.quote_plus(search_term)  # 使用 quote_plus 来编码空格为 "+"，这是 URL 查询参数中常见的格式
        # 构造完整的 URL
        base_url = f'https://www.bbc.com/search?q={encoded_search_term}&page={i}'  # 搜索页面的URL
        if check_url_status(base_url):
            folder = f'result/{search_term}'  # 为每个关键词创建一个独立文件夹
            do_link(base_url, folder)
        else:
            logging.info("URL 访问失败，循环结束")
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
        'public policy', 'social equity', 'income inequality', 'poverty alleviation', 'environmental justice',

        # 更多文化和社会层面
        'intergenerational relationships', 'elderly care', 'urbanization', 'sustainable living', 'intercultural communication',
        'political correctness', 'sustainable development', 'global warming', 'racial equality', 'diversity inclusion',
        'peacekeeping', 'religion and politics', 'disability rights', 'workplace diversity', 'social mobility',
        'urban poverty', 'education reform', 'gender-based violence', 'youth empowerment', 'mental health stigma',
        'voting rights', 'charity and philanthropy', 'economic development', 'housing rights', 'discrimination',
        'international relations', 'social integration', 'collective bargaining', 'affordable healthcare', 'public welfare'
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:  # 限制最大并发线程数为 5
        executor.map(main, search_terms)
