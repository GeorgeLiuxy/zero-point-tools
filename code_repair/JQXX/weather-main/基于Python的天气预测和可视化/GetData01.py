import requests
import pandas as pd
from datetime import datetime

# ====== 用户配置 ======
API_KEY = "f19f3b8390e0482194dcdd749ac7e997"  # 替换成你自己的和风天气 API Key
DEFAULT_LOCATION = "101010100"  # 默认北京，可更换为城市编码

# 支持的预报天数（根据和风天气文档）
ALLOWED_DAYS = [3, 7, 10, 15, 30]


# ====== 主函数：获取天气数据并转为 DataFrame ======
def fetch_weather_forecast(location=DEFAULT_LOCATION, days=3):
    if days not in ALLOWED_DAYS:
        raise ValueError(f"days must be one of {ALLOWED_DAYS}")

    url = f"https://jy3md822t8.re.qweatherapi.com/v7/weather/{days}d"
    params = {
        "location": location,
        "key": API_KEY
    }

    res = requests.get(url, params=params)
    if res.status_code != 200:
        raise Exception("Request failed with HTTP status:", res.status_code)

    data = res.json()
    if data.get("code") != "200":
        raise Exception("API returned error:", data.get("code", "unknown"))

    forecast = data["daily"]
    records = []
    for day in forecast:
        records.append({
            "日期": day["fxDate"],
            "最高温度": day["tempMax"],
            "最低温度": day["tempMin"],
            "天气白天": day["textDay"],
            "天气夜间": day["textNight"],
            "风向": day["windDirDay"],
            "风力": day["windScaleDay"],
            "湿度": day["humidity"],
            "气压": day["pressure"],
            "降水量": day["precip"]
        })

    return pd.DataFrame(records)


# ====== 获取今日天气（从预报中筛选当天） ======
def get_today_weather(location=DEFAULT_LOCATION):
    df = fetch_weather_forecast(location, days=3)
    today = datetime.now().strftime("%Y-%m-%d")
    return df[df["日期"] == today]


# ====== 获取一周天气（调用 7 天预报） ======
def get_week_weather(location=DEFAULT_LOCATION):
    return fetch_weather_forecast(location, days=7)


# ====== 支持全国多个城市调用（如省会城市） ======
def get_china_province_weather_today():
    # 省会城市的和风天气 location code（示例：部分城市）
    city_codes = {
        "北京": "101010100", "上海": "101020100", "广州": "101280101",
        "长春": "101060101", "西安": "101110101", "成都": "101270101"
        # 你可继续添加更多城市
    }

    records = []
    for city, code in city_codes.items():
        df = get_today_weather(code)
        df.insert(0, "城市", city)
        records.append(df)

    return pd.concat(records).reset_index(drop=True)


# ====== 示例调用（可注释） ======
if __name__ == "__main__":
    print("【今日天气】")
    print(get_today_weather())

    print("\n【未来7天天气】")
    print(get_week_weather())

    print("\n【全国省会城市今日天气】")
    print(get_china_province_weather_today())
