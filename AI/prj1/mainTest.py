import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from geopy.distance import geodesic
from tqdm import tqdm
import warnings
import random  # 确保导入 random 模块

# 忽略不相关的警告
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')

# 1. 数据加载
user_actions = pd.read_csv('user_action.csv')
merchants = pd.read_csv('merchant.csv')
test_users = pd.read_csv('testA.csv')

# 2. 数据预处理

# 2.1 处理时间特征
user_actions['action_time'] = pd.to_datetime(user_actions['action_time'])
user_actions['hour'] = user_actions['action_time'].dt.hour
user_actions['day_of_week'] = user_actions['action_time'].dt.dayofweek

# 2.2 过滤点击行为
click_actions = user_actions[user_actions['action'] == 'click'].copy()

# 2.3 标签定义：预测未来点击，标签为1
click_actions['label'] = 1

# 2.4 合并商户信息，避免列名冲突
click_actions = click_actions.merge(merchants, on='mrch_id', how='left', suffixes=('_user', '_mrch'))

# 添加打印语句以验证合并后的列名
print("合并后的 click_actions 列名:", click_actions.columns.tolist())

# 确保 'brand_nm' 和 'mrch_type' 存在
expected_columns = ['brand_nm', 'mrch_type']
missing_columns = [col for col in expected_columns if col not in click_actions.columns]
if missing_columns:
    raise KeyError(f"合并后的数据中缺少列: {missing_columns}")

# 3. 特征工程

# 3.1 用户特征
user_feat = click_actions.groupby('user_id').agg({
    'mrch_id': 'nunique',
    'brand_nm': lambda x: x.mode()[0] if not x.mode().empty else 'unknown',
    'mrch_type': lambda x: x.mode()[0] if not x.mode().empty else 'unknown',
    'latitude_user': 'mean',
    'longitude_user': 'mean',
    'city_id_user': 'first'
}).reset_index().rename(columns={
    'mrch_id': 'unique_merchant_count',
    'brand_nm': 'top_brand',
    'mrch_type': 'top_mrch_type',
    'latitude_user': 'user_latitude',
    'longitude_user': 'user_longitude',
    'city_id_user': 'user_city_id'
})

# 3.2 商户特征
merchant_feat = merchants.copy()
merchant_feat['has_coupon'] = merchant_feat['has_coupon'].astype(int)

# 4. 构建训练样本

# 4.1 正样本
train_pos = click_actions[['user_id', 'mrch_id', 'label']].copy()

# 4.2 负样本（未点击的商户）
def generate_negative_samples(user_id, clicked_mrch_ids, all_mrch_ids, num_samples=10):
    available_mrch_ids = list(set(all_mrch_ids) - set(clicked_mrch_ids))
    if len(available_mrch_ids) == 0:
        return pd.DataFrame()
    sampled_mrch_ids = random.sample(available_mrch_ids, min(num_samples, len(available_mrch_ids)))
    return pd.DataFrame({
        'user_id': [user_id]*len(sampled_mrch_ids),
        'mrch_id': sampled_mrch_ids,
        'label': 0
    })

all_mrch_ids = merchants['mrch_id'].unique()
negative_samples = []
for user in tqdm(click_actions['user_id'].unique(), desc="生成负样本"):
    clicked = click_actions[click_actions['user_id'] == user]['mrch_id'].unique()
    neg = generate_negative_samples(user, clicked, all_mrch_ids, num_samples=10)
    if not neg.empty:
        negative_samples.append(neg)

train_neg = pd.concat(negative_samples, ignore_index=True) if negative_samples else pd.DataFrame(columns=['user_id', 'mrch_id', 'label'])

# 4.3 合并正负样本
train_data = pd.concat([train_pos, train_neg], ignore_index=True)

# 4.4 合并用户特征和商户特征
train_data = train_data.merge(user_feat, on='user_id', how='left')
train_data = train_data.merge(merchant_feat, on='mrch_id', how='left')

# 4.5 处理缺失值
train_data.fillna({'top_brand': 'unknown', 'top_mrch_type': 'unknown'}, inplace=True)

# 4.6 编码类别特征
le_brand = LabelEncoder()
train_data['top_brand_enc'] = le_brand.fit_transform(train_data['top_brand'])

le_type = LabelEncoder()
train_data['top_mrch_type_enc'] = le_type.fit_transform(train_data['top_mrch_type'])

le_brand_mrch = LabelEncoder()
train_data['brand_nm_enc'] = le_brand_mrch.fit_transform(train_data['brand_nm'])

le_type_mrch = LabelEncoder()
train_data['mrch_type_enc'] = le_type_mrch.fit_transform(train_data['mrch_type'])

# 4.7 计算距离特征
def calculate_distance(lat1, lon1, lat2, lon2):
    try:
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers
    except:
        return 0.0

train_data['distance'] = train_data.apply(lambda row: calculate_distance(
    row['user_latitude'], row['user_longitude'], row['latitude'], row['longitude']
), axis=1)

# 4.8 其他特征
train_data['has_coupon'] = train_data['has_coupon'].astype(int)

# 4.9 添加时间特征
# 合并时间特征
train_data = train_data.merge(
    click_actions[['user_id', 'mrch_id', 'hour', 'day_of_week']],
    on=['user_id', 'mrch_id'],
    how='left'
)
train_data['hour'] = train_data['hour'].fillna(0).astype(int)
train_data['day_of_week'] = train_data['day_of_week'].fillna(0).astype(int)

# 5. 特征选择
feature_cols = [
    'unique_merchant_count',
    'top_brand_enc',
    'top_mrch_type_enc',
    'brand_nm_enc',
    'mrch_type_enc',
    'user_latitude',
    'user_longitude',
    'latitude',
    'longitude',
    'distance',
    'has_coupon',
    'hour',
    'day_of_week'
]

# 6. 划分训练集和验证集
X = train_data[feature_cols]
y = train_data['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 7. 模型训练 使用 scikit-learn API
model = LGBMClassifier(
    objective='binary',
    metric='auc',
    boosting_type='gbdt',
    learning_rate=0.05,
    num_leaves=31,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    n_estimators=1000,
    random_state=42
)

# 使用早停
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    # early_stopping_rounds=50,
    # verbose=100
)

# 8. 模型评估
y_pred = model.predict_proba(X_val)[:,1]
print(f'Validation AUC: {roc_auc_score(y_val, y_pred):.4f}')

# 9. 准备推荐

# 9.1 获取所有商户列表
all_merchants = merchants.copy()

# 9.2 为测试用户生成候选商户
# 假设推荐的商户需要在用户所在城市

# 9.3 特征工程为测试集生成特征
test_users = test_users.merge(user_feat, on='user_id', how='left')
test_users.fillna({
    'unique_merchant_count': 0,
    'top_brand': 'unknown',
    'top_mrch_type': 'unknown',
    'user_latitude': test_users['latitude'],
    'user_longitude': test_users['longitude'],
    'user_city_id': test_users['city_id']
}, inplace=True)

# 编码类别特征
test_users['top_brand_enc'] = test_users['top_brand'].map(lambda x: le_brand.transform([x])[0] if x in le_brand.classes_ else -1)
test_users['top_mrch_type_enc'] = test_users['top_mrch_type'].map(lambda x: le_type.transform([x])[0] if x in le_type.classes_ else -1)

# 10. 推荐函数
def generate_recommendations(user, model, top_n=10):
    user_id = user['user_id']
    user_city = user['user_city_id']
    user_lat = user['user_latitude']
    user_lon = user['user_longitude']

    # 获取用户特征
    unique_merchant_count = user['unique_merchant_count']
    top_brand_enc = user['top_brand_enc']
    top_mrch_type_enc = user['top_mrch_type_enc']

    # 获取候选商户
    candidates = all_merchants[all_merchants['city_id'] == user_city].copy()

    # 排除用户已点击的商户
    user_clicked = click_actions[click_actions['user_id'] == user_id]['mrch_id'].unique()
    candidates = candidates[~candidates['mrch_id'].isin(user_clicked)]

    if candidates.empty:
        return []

    # 特征构造
    candidates['unique_merchant_count'] = unique_merchant_count
    candidates['top_brand_enc'] = top_brand_enc
    candidates['top_mrch_type_enc'] = top_mrch_type_enc

    # 编码品牌和类型，处理未知品牌和类型
    candidates['brand_nm_enc'] = candidates['brand_nm'].map(lambda x: le_brand_mrch.transform([x])[0] if x in le_brand_mrch.classes_ else -1)
    candidates['mrch_type_enc'] = candidates['mrch_type'].map(lambda x: le_type_mrch.transform([x])[0] if x in le_type_mrch.classes_ else -1)

    # 计算距离
    candidates['distance'] = candidates.apply(lambda row: calculate_distance(
        user_lat, user_lon, row['latitude'], row['longitude']
    ), axis=1)

    # 其他特征
    candidates['has_coupon'] = candidates['has_coupon'].astype(int)

    # 时间特征，假设当前时间为固定值
    current_hour = 12
    current_day = 3  # 假设为星期三
    candidates['hour'] = current_hour
    candidates['day_of_week'] = current_day

    # 特征选择
    X_candidates = candidates[feature_cols]

    # 处理可能存在的-1编码
    X_candidates = X_candidates.fillna(-1)

    # 预测概率
    scores = model.predict_proba(X_candidates)[:,1]
    candidates['score'] = scores

    # 排序并选取Top N
    top_candidates = candidates.sort_values(by='score', ascending=False).head(top_n)

    return top_candidates['mrch_id'].tolist()

# 11. 生成推荐
recommendations = []
for _, user in tqdm(test_users.iterrows(), total=test_users.shape[0], desc="生成推荐"):
    recs = generate_recommendations(user, model, top_n=10)
    recommendations.append({
        'user_id': user['user_id'],
        'recommended_mrch_ids': ' '.join(map(str, recs))
    })

# 12. 保存推荐结果
recommendations_df = pd.DataFrame(recommendations)
recommendations_df.to_csv('user_recommendations.csv', index=False)

print("推荐结果已保存到 'user_recommendations.csv'。")
