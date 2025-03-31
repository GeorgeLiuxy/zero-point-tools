import requests

# 获取代理节点列表
response = requests.get('http://localhost:9090/proxies')
proxies = response.json()

# 打印代理节点信息
print(proxies)


# curl -X POST -d '{"name": "台湾 A |  | ChatGPT专用"}' http://localhost:9090/proxies/select
