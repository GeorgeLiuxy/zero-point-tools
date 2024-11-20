# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class JobScraperItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    title = scrapy.Field()          # 职位名称
    salary = scrapy.Field()         # 薪资
    salary_evaluation = scrapy.Field()  # 薪酬考核
    requirements = scrapy.Field()    # 职位要求
    company = scrapy.Field()         # 公司信息

