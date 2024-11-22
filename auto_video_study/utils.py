import time

def format_time(seconds):
    """格式化时间为分钟:秒"""
    mins, secs = divmod(seconds, 60)
    return f"{mins:02}:{secs:02}"

def log_message(message):
    """简单的日志输出"""
    print(message)