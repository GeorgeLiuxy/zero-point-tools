import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
from tqdm import tqdm

# 初始化Faker
fake = Faker()

# 设置随机种子以保证可重复性
np.random.seed(42)
random.seed(42)

# 1. 生成城市信息
num_cities = 100
cities = []
for city_id in range(1, num_cities + 1):
    city_name = f"City_{city_id}"
    # 随机生成城市的纬度和经度（假设在中国范围内，大致纬度 18-54，经度 73-135）
    latitude = round(random.uniform(18.0, 54.0), 6)
    longitude = round(random.uniform(73.0, 135.0), 6)
    cities.append({
        'city_id': city_id,
        'city_name': city_name,
        'latitude': latitude,
        'longitude': longitude
    })
cities_df = pd.DataFrame(cities)

# 2. 生成商户信息
num_merchants = 500
brand_names = [f"Brand_{i}" for i in range(1, 51)]  # 50个品牌
merchant_types = ['餐饮', '娱乐', '购物', '教育', '医疗']  # 餐饮、娱乐、购物、教育、医疗

merchants = []
for mrch_id in range(1, num_merchants + 1):
    mrch_nm = f"Merchant_{mrch_id}"
    brand_nm = random.choice(brand_names)
    mrch_type = random.choice(merchant_types)
    city = random.choice(cities)
    city_id = city['city_id']
    # 商户的地理位置略微偏离城市中心
    latitude = round(city['latitude'] + random.uniform(-0.05, 0.05), 6)
    longitude = round(city['longitude'] + random.uniform(-0.05, 0.05), 6)
    has_coupon = random.choice([0, 1])  # 0: 无优惠券, 1: 有优惠券
    merchants.append({
        'mrch_id': mrch_id,
        'mrch_nm': mrch_nm,
        'brand_nm': brand_nm,
        'city_id': city_id,
        'mrch_type': mrch_type,
        'latitude': latitude,
        'longitude': longitude,
        'has_coupon': has_coupon
    })
merchants_df = pd.DataFrame(merchants)
merchants_df.to_csv('merchant.csv', index=False)
print("merchant.csv 已生成。")

# 3. 生成用户信息
num_users = 1000
users = []
for user_id in range(1, num_users + 1):
    city = random.choice(cities)
    city_id = city['city_id']
    # 用户的地理位置略微偏离城市中心
    latitude = round(city['latitude'] + random.uniform(-0.1, 0.1), 6)
    longitude = round(city['longitude'] + random.uniform(-0.1, 0.1), 6)
    users.append({
        'user_id': user_id,
        'city_id': city_id,
        'latitude': latitude,
        'longitude': longitude
    })
users_df = pd.DataFrame(users)
users_df.to_csv('testA.csv', index=False)
print("testA.csv 已生成。")

# 4. 生成用户行为数据
actions = ['browse', 'click', 'purchase']  # 行为类型
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
delta = end_date - start_date

user_actions = []
for user in tqdm(range(1, num_users + 1), desc="生成用户行为数据"):
    # 每个用户随机进行50到200次行为
    num_actions = random.randint(50, 200)
    user_city = users_df.loc[users_df['user_id'] == user, 'city_id'].values[0]
    user_lat = users_df.loc[users_df['user_id'] == user, 'latitude'].values[0]
    user_lon = users_df.loc[users_df['user_id'] == user, 'longitude'].values[0]

    # 为每个用户维护一个品牌偏好列表
    # 初始时为空
    user_brand_prefs = []

    for _ in range(num_actions):
        action = random.choices(actions, weights=[0.6, 0.3, 0.1], k=1)[0]  # 浏览、点击、购买的概率分布
        # 随机生成行为时间
        random_days = random.randint(0, delta.days)
        random_seconds = random.randint(0, 86400)  # 一天中的秒数
        action_time = start_date + timedelta(days=random_days, seconds=random_seconds)

        if action in ['click', 'purchase']:
            # 优先选择用户所在城市的商户
            possible_merchants = merchants_df[merchants_df['city_id'] == user_city]
            if not possible_merchants.empty:
                # 如果用户有品牌偏好，优先选择偏好的品牌
                if user_brand_prefs:
                    # 给予偏好品牌更高的选择概率
                    # 转换为列表以避免索引问题
                    possible_merchants_list = possible_merchants['brand_nm'].tolist()
                    brand_weights = [5 if brand in user_brand_prefs else 1 for brand in possible_merchants_list]
                    chosen_brand = random.choices(possible_merchants_list, weights=brand_weights, k=1)[0]
                    filtered_merchants = possible_merchants[possible_merchants['brand_nm'] == chosen_brand]
                    if not filtered_merchants.empty:
                        mrch = filtered_merchants.sample(1).iloc[0]
                    else:
                        mrch = possible_merchants.sample(1).iloc[0]
                else:
                    mrch = possible_merchants.sample(1).iloc[0]
                mrch_id = mrch['mrch_id']
                # 更新用户品牌偏好
                user_brand_prefs.append(mrch['brand_nm'])
                if len(user_brand_prefs) > 5:
                    user_brand_prefs.pop(0)  # 保持最近5次点击的品牌
            else:
                # 如果所在城市没有商户，则随机选择其他城市的商户
                mrch = merchants_df.sample(1).iloc[0]
                mrch_id = mrch['mrch_id']
        else:
            # 浏览行为可以浏览任意商户
            mrch = merchants_df.sample(1).iloc[0]
            mrch_id = mrch['mrch_id']

        user_actions.append({
            'action': action,
            'user_id': user,
            'action_time': action_time.strftime('%Y-%m-%d %H:%M:%S'),
            'city_id': user_city,
            'mrch_id': mrch_id,
            'latitude': user_lat,
            'longitude': user_lon
        })

user_actions_df = pd.DataFrame(user_actions)
user_actions_df.to_csv('user_action.csv', index=False)
print("user_action.csv 已生成。")
