import os
import re
import requests
from lxml import etree
from requests.exceptions import RequestException
# from multiprocessing import Pool


def get_page(url):
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36                           (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36',
    }
    try:
        response = requests.get(url, headers)
        if response.status_code == 200:
            return response.text
        return None
    except RequestException:
        print('请求出错', url)
        return None


def parse_page_index(html):
    try:
        tree = etree.HTML(html)
        items_xpath = "//div[@class='items']//a/@href"
        items_urls = tree.xpath(items_xpath)
        downloaded_path = f'{os.getcwd()}/downloaded.txt'
        # 检测是否存在已下载url文件
        if not os.path.exists(downloaded_path):
            with open(downloaded_path, 'w') as f:
                f.write('DOWNLOADED URL\n')
        # 获取已下载url列表并转为set
        with open(downloaded_path, 'r') as j:
            downloaded = set(j.readlines())
        if items_urls:
            for item_url in items_urls:
                # 判断item是否已下载
                if item_url not in downloaded:
                    item_name = re.findall('(unit.*)', item_url)[0].replace('/', '-')
                    item_html = get_page(item_url)
                    parse_page_item(item_html, item_name)
                    download_url = re.sub('sess(.*)', 'downloads', item_url)

                    download_html = get_page(download_url)
                    # 获取并格式化session

                    session = item_url.split('/')[-2][1:].replace('-', ' ')
                    parse_page_download(download_html, item_name, session)
                    with open(f'{os.getcwd()}/downloaded.txt', 'a') as k:
                        k.write(f'{item_url}\n')

    except Exception as e:
        raise e


def parse_page_item(html, item_name):
    try:
        tree = etree.HTML(html)
        # 获取item页面文本
        texts_xpath = "//div[contains(@id, 'hide')]/div/div/p//text()"

        texts_list = tree.xpath(texts_xpath)
        if texts_list:
            texts = '\r\n\r\n'.join(texts_list).encode('utf8')
            save_texts(texts, item_name)

    except Exception as e:
        raise e


def parse_page_download(html, item_name, session):
    try:
        tree = etree.HTML(html)
        mp3_xpath = f"//div[contains(text(), '{session}')]/following-sibling::div/a/@href"


        mp3_url = tree.xpath(mp3_xpath)
        if mp3_url:
            mp3_url = mp3_url[0]
            download_mp3(mp3_url, item_name)

    except Exception as e:
        raise e


def download_mp3(url, item_name):
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36                           (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36',
    }
    try:
        response = requests.get(url, headers)
        if response.status_code == 200:
            save_mp3(response.content, item_name)
        return None
    except RequestException:
        print('请求mp3出错', item_name)
        return None


def save_texts(texts, item_name):
    file_path = f'{os.getcwd()}/{item_name}.txt'
    if not os.path.exists(file_path):
        with open(file_path, 'wb') as f:
            f.write(texts)
    

def save_mp3(content, item_name):
    file_path = f'{os.getcwd()}/{item_name}.mp3'
    if not os.path.exists(file_path):
        with open(file_path, 'wb') as f:
            f.write(content)


def bbc_mkdir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    os.chdir(dirname)


def main():
    easy_url = 'http://www.bbc.co.uk/learningenglish/english/basic-grammar'
    medium_url = 'http://www.bbc.co.uk/learningenglish/english/intermediate-grammar'
    easy_html = get_page(easy_url)
    medium_html = get_page(medium_url)
    origin_dir = os.getcwd()
    if easy_html:
        dirname = 'bbc_easy'
        bbc_mkdir(dirname)
        parse_page_index(easy_html)
        os.chdir(origin_dir)
    if medium_url:
        dirname = 'bbc_medium'
        bbc_mkdir(dirname)
        parse_page_index(medium_html)


if __name__ == '__main__':
    main()
    # groups = [x for x in range(5)]
    # pool = Pool()
    # pool.map(main, groups)
