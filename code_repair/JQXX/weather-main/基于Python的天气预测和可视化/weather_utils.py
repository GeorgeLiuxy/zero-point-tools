# weather_utils.py

from datetime import datetime, timedelta
from meteostat import Point, Daily
import pandas as pd

# ✅ 城市名与经纬度映射字典（可自由扩展）
city_coords = {
    "黑龙江": Point(45.75, 126.65), "内蒙古": Point(40.82, 111.65),
    "吉林": Point(43.88, 125.35), "辽宁": Point(41.8, 123.43),
    "河北": Point(38.04, 114.51), "天津": Point(39.12, 117.2),
    "山西": Point(37.87, 112.55), "陕西": Point(34.26, 108.95),
    "甘肃": Point(36.06, 103.83), "宁夏": Point(38.47, 106.27),
    "青海": Point(36.62, 101.78), "新疆": Point(43.82, 87.62),
    "西藏": Point(29.65, 91.13), "四川": Point(30.67, 104.07),
    "重庆": Point(29.56, 106.55), "山东": Point(36.67, 117.0),
    "河南": Point(34.75, 113.62), "江苏": Point(32.07, 118.78),
    "安徽": Point(31.86, 117.28), "湖北": Point(30.58, 114.27),
    "浙江": Point(30.25, 120.17), "福建": Point(26.08, 119.3),
    "江西": Point(28.68, 115.85), "湖南": Point(28.23, 112.93),
    "贵州": Point(26.65, 106.63), "广西": Point(22.82, 108.32),
    "海南": Point(20.03, 110.35), "上海": Point(31.23, 121.47),
    "广东": Point(23.13, 113.27), "云南": Point(25.04, 102.71),
    "台湾": Point(25.03, 121.56), "长春": Point(43.88, 125.35)
}

def get_weather_label(prcp, wspd):
    if prcp > 0:
        return "降水"
    elif wspd > 20:
        return "多风"
    else:
        return "晴/阴"

def enrich_weather_df(df):
    df = df.copy()  # ✅ 避免 SettingWithCopyWarning
    df = df.rename(columns={"tmax": "最高温", "tmin": "最低温", "tavg": "平均温"})
    df["天气"] = df.apply(lambda row: get_weather_label(row["prcp"], row["wspd"]), axis=1)
    df["风力风向"] = df["wspd"].astype(str) + " m/s, " + df["wdir"].astype(str)
    df["空气质量指数"] = df.apply(lambda row: round((1 / row["pres"] * row["wspd"] * 1000), 2) if row["pres"] and row["wspd"] else 0, axis=1)
    df["日期"] = df["time"].dt.strftime("%Y-%m-%d")
    return df[["日期", "平均温", "最低温", "最高温", "prcp", "wspd", "wdir", "pres", "天气", "风力风向", "空气质量指数"]]

# ✅ 获取某城市某月天气（作为 craw_table 替代）
def craw_table(city: str, year: int, month: int) -> pd.DataFrame:
    location = city_coords[city]
    start = datetime(year, month, 1)
    end = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
    data = Daily(location, start, end).fetch().reset_index()
    return enrich_weather_df(data)

# ✅ 获取今日天气
def getToday(city="长春") -> pd.DataFrame:
    today = datetime.now()
    df = craw_table(city, today.year, today.month)
    return df[df["日期"] == today.strftime("%Y-%m-%d")]

# ✅ 获取最近一周天气
def getWeek(city="长春") -> pd.DataFrame:
    location = city_coords[city]
    end = datetime.now()
    start = end - timedelta(days=7)
    data = Daily(location, start, end).fetch().reset_index()
    return enrich_weather_df(data)

# ✅ 获取历史多年月天气（如过去 5 年）
def getYears(city="长春", years=5) -> pd.DataFrame:
    location = city_coords[city]
    today = datetime.now()
    start = datetime(today.year - years, 1, 1)
    data = Daily(location, start, today).fetch().reset_index()
    return enrich_weather_df(data)

# ✅ 获取任意起止日期范围的天气数据
def getPredictDate(year0, month0, day0, year1, month1, day1, city="吉林") -> pd.DataFrame:
    location = city_coords[city]
    start = datetime(year0, month0, day0)
    end = datetime(year1, month1, day1)
    data = Daily(location, start, end).fetch().reset_index()
    return enrich_weather_df(data)

# ✅ 获取全国今日天气（31省会城市）
def getChinaToday() -> pd.DataFrame:
    all_data = []
    end = datetime.now()
    start = end - timedelta(days=2)
    for province, location in city_coords.items():
        data = Daily(location, start, end).fetch().reset_index()
        if not data.empty:
            df = enrich_weather_df(data)
            df = df[df["日期"] == end.strftime("%Y-%m-%d")]
            if not df.empty:
                row = df.iloc[-1]
                row["城市"] = province
                all_data.append(row)
    return pd.DataFrame(all_data)



# ✅ 测试函数
if __name__ == "__main__":
    print("\n>> 今日长春天气:")
    print(getToday("吉林"))

    print("\n>> 近7天吉林天气:")
    print(getWeek("吉林"))

    print("\n>> 吉林近5年天气（尾部预览）:")
    print(getYears("吉林").tail())

    print("\n>> 指定日期段天气 (2024-05-01 至 2024-05-07):")
    print(getPredictDate(2024, 5, 1, 2024, 5, 7, city="吉林"))

    print("\n>> 全国省会城市今日天气:")
    print(getChinaToday().head())
