from PIL import Image, ImageDraw, ImageFont
import barcode
from barcode.writer import ImageWriter

# 字体路径 (macOS，根据实际系统调整)
FONT_PATH = "/System/Library/Fonts/PingFang.ttc"

# 生成条形码图片
def generate_barcode(content, barcode_type='code128', save_path='barcode.png'):
    BARCODE_CLASS = barcode.get_barcode_class(barcode_type)
    bar = BARCODE_CLASS(content, writer=ImageWriter())
    bar.save(save_path)
    return save_path

# 电机标签完整实现
def generate_motor_label_with_barcode(save_path='motor_label_with_barcode.png'):
    width, height = 600, 540
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    font_title = ImageFont.truetype(FONT_PATH, 28)
    font_content = ImageFont.truetype(FONT_PATH, 20)

    # 公司名称与英文
    draw.text((width/2-140, 10), "XXXX电机制造有限公司", font=font_title, fill='black')
    draw.text((width/2-110, 45), "Electric Motor Co.,Ltd", font=font_content, fill='black')
    draw.line([(10,80),(width-10,80)], fill='black', width=2)

    # 完整的电机标签信息
    motor_info = [
        "产品名称: XXXXXXXXXX",
        "型号规格: XXXXXXXX",
        "额定功率: XX kW      额定频率: XX Hz",
        "额定电压: XX V       额定电流: XX A",
        "额定转速: XX r/min  功率因数: XX",
        "绝缘等级: XX           防护等级: XX",
        "接法: XX               环境温度: -XX℃～+XX℃",
        "制造日期: XXXX年XX月",
        "生产许可证号: XK06-123-45678"
    ]

    y = 100
    for line in motor_info:
        draw.text((20, y), line, font=font_content, fill='black')
        y += 30

    draw.line([(10,y+10),(width-10,y+10)], fill='black', width=2)

    # 地址与联系方式
    draw.text((20, y+20), "地址: XXX省XXX市XXX工业园区", font=font_content, fill='black')
    draw.text((20, y+50), "电话: 021-12345678", font=font_content, fill='black')

    # 添加条形码（Code128，内容为电机编号）
    barcode_path = generate_barcode('MOTOR20250328001', 'code128')
    barcode_img = Image.open(barcode_path).resize((400,80))
    img.paste(barcode_img, (100,y+80))

    img.save(save_path)
    print(f"[完成] 电机标签图片生成：{save_path}")

# 食品标签完整实现
def generate_food_label_with_barcode(save_path='food_label_with_barcode.png'):
    width, height = 600, 560
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    font_title = ImageFont.truetype(FONT_PATH, 28)
    font_content = ImageFont.truetype(FONT_PATH, 20)

    # 公司名称与英文
    draw.text((width/2-100, 10), "XX食品有限公司", font=font_title, fill='black')
    draw.text((width/2-90, 45), "XX Food Co.,Ltd", font=font_content, fill='black')
    draw.line([(10,80),(width-10,80)], fill='black', width=2)

    # 完整食品标签信息
    food_info = [
        "产品名称: XXXXXXXXXX",
        "配料: XXXX、XX、XXX、XXX",
        "食品添加剂: XX、XX",
        "净含量: XXXX",
        "保质期: XX个月",
        "贮存条件: XXXXXXXXXXXXXXXX",
        "生产日期: XX年XX月XX日",
        "生产许可证编号: SC12345678901234",
        "产品标准号: GB/T 20980"
    ]

    y = 100
    for line in food_info:
        draw.text((20, y), line, font=font_content, fill='black')
        y += 30

    draw.line([(10,y+10),(width-10,y+10)], fill='black', width=2)

    # 地址与联系方式
    draw.text((20, y+20), "地址: XXX省XXX市XXX食品工业园", font=font_content, fill='black')
    draw.text((20, y+50), "电话: 400-123-4567", font=font_content, fill='black')

    # 添加条形码（EAN13，内容为食品EAN13编号）
    barcode_path = generate_barcode('690123456789', 'ean13')
    barcode_img = Image.open(barcode_path).resize((400,80))
    img.paste(barcode_img, (100,y+80))

    img.save(save_path)
    print(f"[完成] 食品标签图片生成：{save_path}")

# 主程序执行
if __name__ == '__main__':
    generate_motor_label_with_barcode()
    generate_food_label_with_barcode()
