from datetime import datetime

# 数据数组
data = [
    (0, '2024-03-05 13:35:42'),
    (1, '2024-03-06 02:30:34'),
    (2, '2024-03-07 02:26:35'),
    (3, '2024-03-08 02:22:35'),
]

# 转换日期字符串为 datetime 对象
converted_data = [(idx, datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')) for idx, date_str in data]

# 输出结果
for item in converted_data:
    print(item)