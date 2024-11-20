import scrapy

from scapers.job_scraper.job_scraper.items import JobScraperItem


class JobSpiderSpider(scrapy.Spider):
    name = "job_spider"
    allowed_domains = ["example.com"]
    start_urls = ["https://example.com"]

    def parse(self, response):
        # 遍历每个职位链接，点击进入详情页面
        job_links = response.css('.job-listing a::attr(href)').getall()
        for link in job_links:
            yield response.follow(link, callback=self.parse_job_details)

    def parse_job_details(self, response):
        # 实例化 JobItem 对象，存储爬取的数据
        item = JobScraperItem()

        # 提取职位信息
        item['title'] = response.css('.job-title::text').get()
        item['salary'] = response.css('.job-salary::text').get()
        item['salary_evaluation'] = response.css('.salary-evaluation::text').get()  # 根据具体网站选择
        item['requirements'] = response.css('.job-requirements::text').getall()  # 处理多条信息
        item['company'] = response.css('.company-info::text').get()

        yield item