import requests

# 定义请求 URL
url = ("https://api.anjuke.com/weixin/broker/ldt?cid=11&from=weapp&app=a-ajk&platform=mac"
       "&b=MacBookPro17,1&s=macOS14.0.0&cv=5.0&wcv=5.0&t=1730700128&wv=7.0.8"
       "&sv=2.18.1&batteryLevel=0&muid=760d051849f22f26bf9a071a72b49132"
       "&weapp_version=1.0.0&user_id=&oid=ocS7q0LJ_R--7-Tk3UuUdHXU-EGU"
       "&udid=o9PQht4RX102OBWQM_VfH22aXcfI&mobile="
       "&type=1&uid=200761893&lego_info=%5Bobject%20Object%5D&bid=62&site=1&city_id=11")

# 定义请求头部
headers = {
    "method": "GET",
    "scheme": "https",
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
    "sig": "835490735756a0d43f49815b918d0f66",
    "ajkauthticket": "",
    "cookie": "CURRENT_VERSION=5.0;aQQ_ajkguid=ocS7q0LJ_R--7-Tk3UuUdHXU-EGU;ajkAuthTicket="
}

# 发送 GET 请求
response = requests.get(url, headers=headers)

# 检查响应状态码
if response.status_code == 200:
    # 如果请求成功，解析并打印 JSON 数据
    data = response.json()
    print("请求成功，返回的数据为：")
    print(data)
else:
    # 如果请求失败，输出状态码和错误信息
    print(f"请求失败，状态码：{response.status_code}")
    print(response.text)
