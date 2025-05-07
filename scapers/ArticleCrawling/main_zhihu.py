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
        base_url = f'https://www.zhihu.com/search?type=content&q={encoded_search_term}&page={i}'  # 搜索页面的URL
        if check_url_status(base_url):
            folder = f'result/{search_term}'  # 为每个关键词创建一个独立文件夹
            do_link(base_url, folder)
        else:
            logging.info("URL 访问失败，循环结束")
            break


if __name__ == "__main__":
    search_terms = [
        # 社交礼仪
        '社交礼仪', '礼仪规范', '社交技巧', '交际礼仪', '座次礼仪', '手势与肢体语言', '商务礼仪',
        '用餐礼仪', '接待礼仪', '尊重个人空间', '恭敬与礼貌', '礼品赠送与回赠', '交际场合注意事项',
        '社交场合的着装要求', '公共场合行为规范', '社交网络礼仪', '职场社交技巧', '网络社交礼仪',

        # 法律制度
        '法律体系', '法律法规', '法律程序', '司法制度', '公平与正义', '法治社会', '个人权利保护',
        '民法', '刑法', '国际法', '法律责任', '法律执行', '合同法', '知识产权法', '劳动法',
        '婚姻法', '家庭法', '商法', '宪法', '国家安全法', '刑事诉讼法', '民事诉讼法', '行政法',
        '法庭审判', '法律援助', '法律实践', '法律教育', '人权', '公民权利', '社会保障法',
        '公司法', '税法', '金融法', '环保法', '公共政策', '移民法', '移民与难民权利',

        # 日常生活
        '日常生活习惯', '生活小窍门', '生活技巧', '家务管理', '健康生活方式', '环保生活',
        '生活成本管理', '家庭预算', '时间管理', '心理健康', '个人理财', '家庭关系', '工作与生活平衡',
        '购物与消费习惯', '现代生活方式', '社交技巧', '饮食习惯', '节俭与消费', '自我提升',
        '健康饮食', '运动健身', '心理调适', '幸福家庭', '环保意识', '生活态度', '家居整理',
        '节能减排', '职场生活', '网络购物', '旅行习惯', '交通出行', '理财规划', '养生习惯',

        # 社会习俗
        '传统习俗', '文化差异', '社会行为规范', '民俗活动', '节日习俗', '婚姻与家庭习俗',
        '生日与纪念日庆祝', '礼节与传统', '社交活动', '社会道德', '公共礼仪', '地区文化差异',
        '婚礼与葬礼习惯', '文化习俗', '节庆活动', '仪式与礼仪', '宗教礼仪', '家庭聚会习惯',
        '老年人习惯', '青少年行为规范', '儿童教育习惯', '民族风俗', '地方传统', '旅行习俗',
        '传统节日庆祝', '跨文化交流', '食物与饮品文化', '社会期望与行为', '公民义务', '道德规范',
        '家庭责任', '职场礼仪', '公共场所行为规范', '集体活动', '民间故事与传说', '社会服务',
        '集体主义与个人主义', '国际节庆习俗', '家庭责任与义务', '社会贡献'
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:  # 限制最大并发线程数为 5
        executor.map(main, search_terms)
