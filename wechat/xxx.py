import time
import requests
import hashlib

def generate_timestamp():
    """生成当前 Unix 时间戳（秒）"""
    return int(time.time())

def generate_sig(parameters, secret_key):
    """
    生成签名（假设签名是参数按字典顺序拼接再加密），
    具体逻辑视API实际签名规则而定
    """
    # 将参数按字母顺序排序
    sorted_params = sorted(parameters.items())
    # 拼接参数
    concatenated_params = ''.join(f"{key}={value}" for key, value in sorted_params)
    # 假设签名生成是 md5 加密
    raw_string = concatenated_params + secret_key  # 加入密钥或其他固定值
    return hashlib.md5(raw_string.encode()).hexdigest()

def fetch_data(page, secret_key):
    # 基本的请求 URL 和参数
    base_url = "https://miniapp.anjuke.com/weapp/broker/list"

    # 请求的 query 参数
    params = {
        "cid": 11,
        "from": "weapp",
        "app": "a-ajk",
        "platform": "mac",
        "b": "MacBookPro17,1",
        "s": "macOS14.0.0",
        "cv": "5.0",
        "wcv": "5.0",
        "t": generate_timestamp(),  # 动态生成时间戳
        "wv": "7.0.8",
        "sv": "2.18.1",
        "batteryLevel": 0,
        "muid": "760d051849f22f26bf9a071a72b49132",
        "weapp_version": "1.0.0",
        "user_id": "undefined",
        "oid": "ocS7q0LJ_R--7-Tk3UuUdHXU-EGU",
        "udid": "o9PQht4RX102OBWQM_VfH22aXcfI",
        "page": page,  # 动态设置页数
        "page_size": 100,
        "city_id": 11,
        "area_id": "",
        "shangquan_id": ""
    }

    # 生成签名并加入到请求参数
    params["sig"] = generate_sig(params, secret_key)

    # 请求头设置
    headers = {
        "content-type": "application/x-www-form-urlencoded",
        "ft": "ajk-weapp",
        "accept": "*/*",
        "switchrecommend": "1",
        "ppu": "",
        "x_ajk_app": "a-weapp",
        "accept-language": "zh-CN,zh-Hans;q=0.9",
        "ak": "f8d5b75d0b3991fbcaf64de096d025537fb40f3e",
        "accept-encoding": "gzip, deflate, br",
        "user-agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 11_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E217 MicroMessenger/6.8.0(0x16080000) NetType/WIFI Language/en Branch/Br_trunk MiniProgramEnv/Mac",
        "referer": "https://servicewechat.com/wx099e0647f9a4717d/1009/page-frame.html",
        "x-forwarded-for": "112.64.131.100",
        "sig": params["sig"],
        "ajkauthticket": "",  # 如果有需要，填写有效的 ajkauthticket
        "cookie": "CURRENT_VERSION=5.0;aQQ_ajkguid=ocS7q0LJ_R--7-Tk3UuUdHXU-EGU;ajkAuthTicket="
    }

    # 发起请求
    response = requests.get(base_url, headers=headers, params=params)
    return response.json()

# 使用示例
secret_key = "your_secret_key"  # 替换为实际的密钥
page = 1  # 请求的页数
try:
    response_data = fetch_data(page, secret_key)
    print(response_data)
except requests.RequestException as e:
    print("请求失败:", e)
