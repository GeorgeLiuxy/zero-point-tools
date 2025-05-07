import joblib
import datetime as DT

from meteostat import Point
import GetModel
import ProcessData
import weather_utils

from pyecharts.charts import Bar, Grid, Line, Tab, Map
from pyecharts.components import Table
from pyecharts.options import ComponentTitleOpts
from pyecharts import options as opts

# 定义城市经纬度映射
def get_province_coords():
    return {
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
        "台湾": Point(25.03, 121.56)
    }

# 加载模型并预测未来7天
r = GetModel.getModel()
print("MAE:", r[0])
model = joblib.load('Model.pkl')
preds = model.predict(r[1])

# 构造预测数据
today = DT.datetime.now()
x_data = [(today + DT.timedelta(days=i)).date() for i in range(7)]
predict_high_temperature = [round(p[0], 2) for p in preds]
predict_low_temperature = [round(p[1], 2) for p in preds]
predict_airs = [round(p[2], 2) for p in preds]

# 今日长春天气表格
def table_main():
    data = weather_utils.getToday("长春")
    headers = ["日期", "最高温", "最低温", "天气", "风力风向", "空气质量指数"]
    row = [[
        data.iloc[0].get("日期", "N/A"),
        data.iloc[0].get("最高温", "N/A"),
        data.iloc[0].get("最低温", "N/A"),
        data.iloc[0].get("天气", "N/A"),
        data.iloc[0].get("风力风向", "N/A"),
        data.iloc[0].get("空气质量指数", "N/A")
    ]]
    return Table().add(headers, row).set_global_opts(
        title_opts=ComponentTitleOpts(title="今日长春天气", subtitle="")
    )

# 近一周长春天气可视化
def grid_week():
    data = weather_utils.getWeek("吉林")
    x = ["前七天", "前六天", "前五天", "前四天", "前三天", "前两天", "前一天"]
    high = ProcessData.setHighTemp(data)
    low = ProcessData.setLowTemp(data)
    air = ProcessData.setAir(data)

    bar = Bar()
    bar.add_xaxis(x)
    bar.add_yaxis("最高温", high, yaxis_index=0, color="#d14a61")

    # 添加第二和第三个 y 轴
    bar.extend_axis(yaxis=opts.AxisOpts(name="最低温", position="right"))
    bar.extend_axis(yaxis=opts.AxisOpts(name="空气质量", position="right", offset=60))

    bar.add_yaxis("最低温", low, yaxis_index=1, color="#5793f3")

    line = Line()
    line.add_xaxis(x)
    line.add_yaxis("空气质量", air, yaxis_index=2, color="#675bba", label_opts=opts.LabelOpts(is_show=False))

    bar.overlap(line)

    grid = Grid()
    grid.add(bar, opts.GridOpts(pos_left="5%", pos_right="20%"), is_control_axis_index=True)
    return grid


# 预测图表
def grid_week_predict():
    bar = Bar()
    bar.add_xaxis([str(x) for x in x_data])
    bar.add_yaxis("最高温", predict_high_temperature, yaxis_index=0, color="#d14a61")

    bar.extend_axis(yaxis=opts.AxisOpts(name="最低温", position="right"))
    bar.extend_axis(yaxis=opts.AxisOpts(name="空气质量", position="right", offset=60))

    bar.add_yaxis("最低温", predict_low_temperature, yaxis_index=1, color="#5793f3")

    line = Line()
    line.add_xaxis([str(x) for x in x_data])
    line.add_yaxis("空气质量", predict_airs, yaxis_index=2, color="#675bba", label_opts=opts.LabelOpts(is_show=False))

    bar.overlap(line)

    grid = Grid()
    grid.add(bar, opts.GridOpts(pos_left="5%", pos_right="20%"), is_control_axis_index=True)
    return grid


# 全国天气表格
def today_china_table(province_coords, df):
    rows = []
    for province in province_coords.keys():
        row_data = df[df["城市"] == province]
        if not row_data.empty:
            row = row_data.iloc[0]
            rows.append([
                province,
                row.get("最低温", "N/A"),
                row.get("最高温", "N/A"),
                row.get("天气", "N/A"),
                row.get("风力风向", "N/A")
            ])
        else:
            rows.append([province, "N/A", "N/A", "N/A", "N/A"])
    return Table().add(["省份", "最低温", "最高温", "天气", "风力风向"], rows).set_global_opts(
        title_opts=ComponentTitleOpts(title="今日全国天气", subtitle="")
    )

# 全国空气质量图
def today_china_map(province_coords, df):
    data = []
    for province in province_coords.keys():
        row_data = df[df["城市"] == province]
        if not row_data.empty:
            aqi = row_data.iloc[0].get("空气质量指数", 0)
            data.append([province, aqi])
        else:
            data.append([province, 0])
    return Map().add("空气质量指数", data, "china").set_global_opts(
        title_opts=opts.TitleOpts(title="全国空气质量"),
        visualmap_opts=opts.VisualMapOpts(max_=300)
    )


# 主函数
province_coords = get_province_coords()
china_today = weather_utils.getChinaToday()
china_today.to_csv("china_today.csv")

# 绘图页面组合
tab = Tab()
tab.add(table_main(), "今日长春")
tab.add(grid_week_predict(), "未来长春")
tab.add(grid_week(), "近一周长春")
tab.add(today_china_table(province_coords, china_today), "今日中国天气")
tab.add(today_china_map(province_coords, china_today), "全国空气质量")
tab.render("天气网.html")
