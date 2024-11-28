from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import base64

# 加密时使用的密钥和加密后的数据
secret_key = "stu_card_1847521".encode('utf-8')  # 密钥，必须是16、24或32字节
encrypted_time_length = "iHc56LteLbuDwl3zyV++RQ=="

# 解密步骤
# Step 1: 对 Base64 编码的加密字符串进行解码
encrypted_data = base64.b64decode(encrypted_time_length)

# Step 2: 创建 AES 解密器，使用 ECB 模式
cipher = AES.new(secret_key, AES.MODE_ECB)

# Step 3: 解密并去除填充
decrypted_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)

# Step 4: 转换解密后的字节数据为整数（原始播放时间）
original_time = int(decrypted_data.decode('utf-8'))

print("解密后的播放时间:", original_time)

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import base64

# 密钥和明文
secret_key = "stu_card_1847521".encode('utf-8')  # 16字节的密钥
current_time = 2*60+56  # 假设这是你要加密的播放时间
print("原始播放时间:", current_time)

# Step 1: 将明文转换为字节
time_bytes = str(current_time).encode('utf-8')  # 转换为字节流

# Step 2: 创建 AES 加密器，使用 ECB 模式
cipher = AES.new(secret_key, AES.MODE_ECB)

# Step 3: 对明文进行填充，使其长度成为 AES 块大小的倍数
padded_data = pad(time_bytes, AES.block_size)

# Step 4: 加密
encrypted_data = cipher.encrypt(padded_data)

# Step 5: 将加密后的数据转换为 Base64 编码
encrypted_base64 = base64.b64encode(encrypted_data).decode('utf-8')

print("加密后的数据:", encrypted_base64)



import base64

# username: MTU1NDAwMzk3NzE=
# password: QWExMjM0NTY3ODk=

def encode_base64(data):
    # 将字符串编码为字节类型，然后进行 Base64 编码
    encoded_bytes = base64.b64encode(data.encode('utf-8'))
    # 将编码后的字节转换回字符串
    return encoded_bytes.decode('utf-8')

# 示例数据
new_password = "15540039771"
re_password = "Aa123456789"

# 编码
encoded_new_password = encode_base64(new_password)
encoded_re_password = encode_base64(re_password)

# 打印结果
print(f"Encoded New Password: {encoded_new_password}")
print(f"Encoded Re Password: {encoded_re_password}")
