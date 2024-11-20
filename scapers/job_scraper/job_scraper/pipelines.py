import sqlite3
from scrapy.exceptions import DropItem


class JobScraperPipeline:
    def open_spider(self, spider):
        self.conn = sqlite3.connect('jobs.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                salary TEXT,
                salary_evaluation TEXT,
                requirements TEXT,
                company TEXT
            )
        ''')

    def close_spider(self, spider):
        self.conn.commit()
        self.conn.close()

    def process_item(self, item, spider):
        # 仅存储与“招商经理”相关的职位
        if "招商经理" not in item['title']:
            raise DropItem(f"Dropped job not related to 招商经理: {item['title']}")

        # 过滤职位要求中不含“管理”或“考核”的职位
        if not any(keyword in item['requirements'] for keyword in ["管理", "考核"]):
            raise DropItem(f"Dropped job missing key requirements: {item['requirements']}")

        # 存储数据
        self.cursor.execute('''
            INSERT INTO jobs (title, salary, salary_evaluation, requirements, company)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            item['title'],
            item['salary'],
            item['salary_evaluation'],
            '\n'.join(item['requirements']),
            item['company']
        ))
        self.conn.commit()
        return item
