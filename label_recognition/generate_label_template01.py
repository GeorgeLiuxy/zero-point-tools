from PIL import Image, ImageDraw, ImageFont
import barcode
from barcode.writer import ImageWriter

FONT_PATH = "/System/Library/Fonts/PingFang.ttc"  # 替换为你系统的字体路径

def generate_motor_label_table_template(save_path='motor_label_table.png'):
    width, height = 600, 700
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    font_title = ImageFont.truetype(FONT_PATH, 26)
    font_cell = ImageFont.truetype(FONT_PATH, 20)

    # 标题
    draw.text((width//2 - 100, 10), "电机铭牌模板", font=font_title, fill='black')

    # 表格字段
    label_data = {
        "产品名称": "三相异步电动机",
        "型号规格": "Y2-112M-4",
        "额定功率": "4.0 kW",
        "额定频率": "50 Hz",
        "额定电压": "380 V",
        "额定电流": "8.6 A",
        "额定转速": "1440 r/min",
        "功率因数": "0.84",
        "绝缘等级": "B",
        "防护等级": "IP54",
        "接法": "△",
        "环境温度": "-15℃～+40℃",
        "制造日期": "2025年03月",
        "许可证号": "XK06-123-45678"
    }

    # 表格样式定义
    start_x, start_y = 30, 60
    col1_width = 150
    col2_width = 400
    row_height = 35

    # 存储字段区域位置
    template_regions = {}

    # 表格绘制
    y = start_y
    for key, value in label_data.items():
        # 绘制网格线
        draw.rectangle([(start_x, y), (start_x + col1_width, y + row_height)], outline='black', width=1)
        draw.rectangle([(start_x + col1_width, y), (start_x + col1_width + col2_width, y + row_height)], outline='black', width=1)

        # 写入内容
        draw.text((start_x + 10, y + 7), key, font=font_cell, fill='black')
        draw.text((start_x + col1_width + 10, y + 7), value, font=font_cell, fill='black')

        # 记录区域
        template_regions[key] = {
            "param_cell": ((start_x, y), (start_x + col1_width, y + row_height)),
            "value_cell": ((start_x + col1_width, y), (start_x + col1_width + col2_width, y + row_height))
        }

        y += row_height

    # 添加条形码（可选）
    barcode_path = generate_barcode('MOTOR20250328001', 'code128')
    barcode_img = Image.open(barcode_path).resize((350, 80))
    img.paste(barcode_img, (120, y + 20))

    # 保存图片
    img.save(save_path)
    print(f"[完成] 表格样式电机标签模板已生成：{save_path}")
    return template_regions

# 生成条形码
def generate_barcode(content, barcode_type='code128', save_path='barcode'):
    BARCODE_CLASS = barcode.get_barcode_class(barcode_type)
    bar = BARCODE_CLASS(content, writer=ImageWriter())
    filename = bar.save(save_path)
    return f"{filename}.png" if not filename.endswith('.png') else filename

# 运行主函数
if __name__ == '__main__':
    regions = generate_motor_label_table_template()
    print("\n模板字段区域坐标：")
    for k, v in regions.items():
        print(f"{k}: {v}")
