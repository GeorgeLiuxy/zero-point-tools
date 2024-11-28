import requests

url = "https://peixunapi.tlsjyy.com.cn/api/course/end_study"
headers = {
    "authorization": "eyJ0eXAiOiJKV1QiLCJhbGciOiJTSEEyNTYifQ.eyJpc3MiOiJTdXBfenpocSIsImlhdCI6MTczMjY5MzEwNiwiZXhwIjoxNzMzMjk3OTA2LCJ1c2VyX2lkIjo1Mjk3MiwidHlwZSI6MSwibW9iaWxlIjoiMTU1NDAwMzk3NzEifQ.8b489329f7b987c3e25e1c6931d7739ab295006133f071ed532974acac5d60e0",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "accept": "application/json, text/plain, */*",
    "sec-ch-ua": "\"Google Chrome\";v=\"131\", \"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\"",
    "content-type": "application/json;charset=UTF-8",
    "sec-ch-ua-mobile": "?0",
    "origin": "https://peixun.tlsjyy.com.cn",
    "sec-fetch-site": "same-site",
    "sec-fetch-mode": "cors",
    "sec-fetch-dest": "empty",
    "referer": "https://peixun.tlsjyy.com.cn/",
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "zh-CN,zh;q=0.9",
    "priority": "u=1, i"
}

data = {
    "chapter_id": 1594,
    "train_id": 1701,
    "time_length": "jHVfMKTppEFpjpZoVzwYOg=="
}

# 禁用代理，直接发请求
response = requests.post(url, headers=headers, json=data, proxies={})

print("响应状态码:", response.status_code)
print("响应内容:", response.json())
