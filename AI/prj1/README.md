# LightGBM 商户推荐实验

该项目根据用户行为、商户和地理位置等信息训练 LightGBM 排序/分类模型，为每个用户生成 Top-N 商户推荐结果。

## 核心流程

1. 读取用户行为、商户和待预测用户数据。
2. 构造品牌偏好、行为类型、优惠券、时间和用户—商户地理距离等特征。
3. 生成正负样本并训练 LightGBM 模型。
4. 使用 AUC 等指标评估，并按预测得分输出 Top-N 推荐。

## 文件

- `mainTest.py`：完整训练、评估与推荐实验入口。
- `data_generate.py`：样例数据生成/整理脚本。
- `user_action.csv`：用户行为样例。
- `merchant.csv`：商户信息样例。
- `testA.csv`：待推荐用户样例。

## 环境与运行

```bash
pip install pandas numpy lightgbm scikit-learn geopy tqdm
cd AI/prj1
python mainTest.py
```

脚本依赖本目录下的 CSV 文件，建议从该目录运行。更换数据时请保持字段名称、经纬度格式、用户/商户 ID 和时间字段与代码一致。

## 输出与限制

结果包括验证指标和每个用户的推荐商户列表。当前实现是离线实验，负采样策略、特征统计范围和训练/验证切分都需要按真实业务重新设计，以避免数据泄漏并保证线上时效性。
