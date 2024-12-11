import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

# 初始化 Faker
fake = Faker(locale='zh_CN')  # 设置为中文环境

# 设置随机种子以确保结果可复现
random.seed(42)
np.random.seed(42)

# 定义2018年至2023年的所有月份
years = range(2018, 2024)  # 包含2023年
months = list(range(1, 13))  # 1到12月

# 初始化空的列表来存储所有订单数据
all_orders = []

# 生成每月的数据
order_id_counter = 1  # 用于生成唯一的订单编号

# 定义特殊月份和对应的订单数量增长因子
special_months = {
    1: {"name": "春节", "order_multiplier": 1.5, "refund_multiplier": 1.2},  # 春节活动
    6: {"name": "618", "order_multiplier": 2.0, "refund_multiplier": 1.2},   # 618大促
    11: {"name": "双十一", "order_multiplier": 3.0, "refund_multiplier": 1.3},  # 双十一大促
    12: {"name": "双十二", "order_multiplier": 2.5, "refund_multiplier": 1.3},  # 双十二大促
}

# 定义每年的春节日期（简化为每年2月，实际可根据农历调整）
spring_festival_month = 2

# 商品种类及价格区间
product_categories = {
    "手机": (1000, 10000),
    "电视": (2000, 20000),
    "家电": (500, 5000),
    "服装": (50, 2000),
    "食品": (10, 500),
    "书籍": (30, 300),
}

# 用户等级定义（VIP 用户更可能购买高价商品）
user_levels = ["普通用户", "VIP用户", "超级VIP用户"]

for year in years:
    for month in months:
        # 判断是否为特殊月份
        if month in special_months:
            order_multiplier = special_months[month]["order_multiplier"]
            refund_multiplier = special_months[month]["refund_multiplier"]
            special = True
            special_name = special_months[month]["name"]
        elif month == spring_festival_month:
            # 春节月份
            order_multiplier = 1.5
            refund_multiplier = 1.2
            special = True
            special_name = "春节"
        else:
            order_multiplier = 1.0
            refund_multiplier = 1.0
            special = False
            special_name = ""

        # 基础订单数量（3000到5000）乘以订单因子
        base_num_orders = random.randint(3000, 5000)
        num_orders = int(base_num_orders * order_multiplier)

        print(f"生成 {year}年{month}月的 {num_orders} 笔订单{' - ' + special_name if special else ''}...")

        for _ in range(num_orders):
            # 生成唯一的订单编号
            order_id = f"ORD{year}{month:02d}{order_id_counter:06d}"
            order_id_counter += 1

            # 随机选择商品类别和价格
            product_category = random.choice(list(product_categories.keys()))
            min_price, max_price = product_categories[product_category]
            total_amount = round(random.uniform(min_price, max_price), 2)

            # 根据用户等级调整订单金额
            user_level = random.choice(user_levels)
            if user_level == "VIP用户":
                total_amount *= 1.2  # VIP用户多买一些高价商品
            elif user_level == "超级VIP用户":
                total_amount *= 1.5  # 超级VIP用户多买高价商品

            # 买家实际支付金额（假设打了折扣，介于总金额的90%到100%之间）
            actual_payment = round(total_amount * random.uniform(0.9, 1.0), 2)

            # 生成随机收货地址
            shipping_address = fake.address().replace("\n", " ")

            # 生成订单创建时间和付款时间
            first_day = datetime(year, month, 1)
            if month == 12:
                next_month = datetime(year + 1, 1, 1)
            else:
                next_month = datetime(year, month + 1, 1)
            last_day = next_month - timedelta(days=1)

            # 生成订单创建时间
            order_creation_time = fake.date_time_between(start_date=first_day, end_date=last_day)

            # 生成订单付款时间，确保晚于创建时间
            order_payment_time = fake.date_time_between(start_date=order_creation_time, end_date=last_day)

            # 退款金额
            base_refund_probability = 0.05
            refund_probability = base_refund_probability * refund_multiplier

            if random.random() < refund_probability:
                # 退款金额在0到实际支付金额的50%之间
                refund_amount = round(random.uniform(0, actual_payment * 0.5), 2)
            else:
                refund_amount = 0.00

            # 将订单数据添加到列表中
            all_orders.append({
                "订单编号": order_id,
                "总金额": total_amount,
                "买家实际支付金额": actual_payment,
                "收货地址": shipping_address,
                "订单创建时间": order_creation_time.strftime("%Y/%m/%d %H:%M"),
                "订单付款时间": order_payment_time.strftime("%Y/%m/%d %H:%M"),
                "退款金额": refund_amount,
                "用户等级": user_level
            })

# 创建 DataFrame
df = pd.DataFrame(all_orders)

# 查看生成的数据
print("\n生成的数据预览：")
print(df.head())

# 保存为 CSV 文件
output_file = "orders_2018_2023_v3.csv"
df.to_csv(output_file, index=False, encoding='utf-8-sig')  # 使用 utf-8-sig 以确保Excel可以正确读取中文

print(f"\n数据已成功保存到 {output_file}")
