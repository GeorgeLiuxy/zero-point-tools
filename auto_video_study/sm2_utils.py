import base64
import json


# 1. 原始 JSON 字符串
json_str = '{"train_id":1701,"type":1}'

# 2. 将 JSON 字符串转换为 UTF-8 字节流
utf8_bytes = json_str.encode('utf-8')

# 3. 对字节流进行 Base64 编码
base64_str = base64.b64encode(utf8_bytes).decode('utf-8')

# 输出 Base64 编码后的字符串
print("Base64 编码后的字符串:", base64_str)

# Base64 编码的字符串
base64_str = "eyJjb3Vyc2VfaWQiOjE5OCwidHJhaW5faWQiOjE3MDF9"

# 1. 对 Base64 编码的字符串进行解码
base64_bytes = base64.b64decode(base64_str)

# 2. 解码后的字节转换为 UTF-8 字符串  "{\"train_id\":1701,\"type\":1}"
decoded_str = base64_bytes.decode('utf-8')

# 3. 将解码后的字符串转换为 JSON 对象
decoded_json = json.loads(decoded_str)

# 输出解码后的 JSON 数据
print("解码后的 JSON 字符串:", decoded_str)
print("转换为字典后的 JSON 数据:", decoded_json)