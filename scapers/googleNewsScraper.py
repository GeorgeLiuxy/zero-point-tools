# 匯入scraparazzie模組
from scraparazzie import scraparazzie

# 建立scraparazzie新聞物件 (新聞物件=client)
""" # scraparazzie物件參數
language 預設english
location 預設United States
topic
query 包含的文字
max_results 爬取的最大數量，預設5，最大可設100 """
client = scraparazzie.NewsClient(language='chiense traditional', location='Taiwan', topic='Business', max_results=3)
client.print_news() # 包含新聞標題、連結、來源機構、發布時間

# 查詢可用參數值
client.languages
client.locations
client.topics
"""['Top Stories', 'World', 'Nation', 'Business', 'Technology', 
'Entertainment', 'Sports', 'Science', 'Health'] """