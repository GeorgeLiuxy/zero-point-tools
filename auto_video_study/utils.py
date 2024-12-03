import base64
import time

def format_time(seconds):
    """格式化时间为分钟:秒"""
    mins, secs = divmod(seconds, 60)
    return f"{mins:02}:{secs:02}"

def log_message(message):
    """简单的日志输出"""
    print(message)


def base64_encode(data):
    """将字符串转换为 Base64 编码"""
    return base64.b64encode(data.encode()).decode('utf-8')

# # 1. 原始 JSON 字符串
# json_str = '{"train_id":1701,"type":1}'
#
# # 2. 将 JSON 字符串转换为 UTF-8 字节流
# utf8_bytes = json_str.encode('utf-8')
#
# # 3. 对字节流进行 Base64 编码
# base64_str = base64.b64encode(utf8_bytes).decode('utf-8')
#
# # 输出 Base64 编码后的字符串
# print("Base64 编码后的字符串:", base64_str)