# import urllib.parse
#
# # 原始 URL 参数 (已经 URL 编码)
# param = "M6FUERTovrwJKhmQd7T7TyNBK%2BVNxhzqSH0PXmorPcZ3rSPEh3uIvka%2FyE3InQ%2F4DeUOXdLAPuFPbuWqM7gto7EJzxuiNJ27XTKne2kfX8TwXayEpRR%2BBx2qNm6PaNCEr%2BBPVXQwETF1E3ZrXKWbu1cvpDZ9wd5SUnjTXA7pyDuLcWXcQ8GtYFczKPc%2Bg%2F%2B5aN4IBNkua1%2Be6miOX0N5twDFySexaWh9Tk1ODzvkOA3oJNVqGZr%2FVxuGbQr9cvLZP0QP409UzNNJrHPUSiT%2FtBGdWQi7FoaUzYfzyYz3wSmJQXQ7OHs1B7UxH9wS%2B0ULVc%2F2DJAdQy2datPAwweQd%2BdQ0YG9vcrTdVWVCMm37v7mT5XxNNTuV4yunhyRGhRcVdBSoF6oP%2F49x7XPnVcGDnZ5CAHdxjc5IARkxOo5ZpTpWC3FGJsslidoK5%2Fe2%2BvSXLSSyoq%2Bev8B0XcmCdbMISGKxyX%2BFnmpAmWq7CDIRwQrlaq%2BFdQBZM4X05Pez1Oh2UVSTT%2BuVQxgeuUDMzDCKeAEz1HQYga90gLFjQEADskcqhkGytZZTipShduFZAXy1dUq3Oof7hc%2FESoi3v2tWDCUo4rkjMh6Mn7THqYG0ev1M5dGGRbtRhs6Fbd%2BBHm%2FmTQ2QJn13O3%2BLedGU8%2B5rLGFA4uP52l3vpMYFpXBzud24QExFgUnmsfjzggtZH0r"
#
# # URL 解码
# decoded_param = urllib.parse.unquote(param)
# print(decoded_param)

import subprocess
import json

def call_js_encryption(data):
    # 将数据转换为 JSON 格式的字符串
    data_str = json.dumps(data)

    # 调用 Node.js 脚本
    result = subprocess.run(
        ['node', './js/index.js', data_str],  # 传入 JSON 字符串作为参数
        capture_output=True,  # 捕获输出
        text=True  # 获取字符串输出
    )

    # 获取并返回加密后的结果
    if result.returncode == 0:
        return result.stdout.strip()  # 返回加密结果
    else:
        raise Exception(f"Error in Node.js: {result.stderr}")

# 示例数据
data = {"train_id": 1701, "type": 1}

# 调用加密方法
encrypted_data = call_js_encryption(data)
print(f"Encrypted Data: {encrypted_data}")
