# #!/usr/bin/env python3
# import sys
# import time
# import jwt
#
# # Open PEM
# private_key = "f19f3b8390e0482194dcdd749ac7e997"
#
# payload = {
#     'iat': int(time.time()) - 30,
#     'exp': int(time.time()) + 900,
#     'sub': 'YOUR_PROJECT_ID'
# }
# headers = {
#     'kid': 'f19f3b8390e0482194dcdd749ac7e997'
# }
#
# # Generate JWT
# encoded_jwt = jwt.encode(payload, private_key, algorithm='EdDSA', headers = headers)
#
# print(f"JWT:  {encoded_jwt}")

# curl --compressed \
#     'https:/jy3md822t8.re.qweatherapi.com/v7/weather/3d?location=101010100&key=f19f3b8390e0482194dcdd749ac7e997'

from meteostat import Point, Daily
from datetime import datetime

location = Point(43.88, 125.35)  # 长春经纬度
start = datetime(2023, 1, 1)
end = datetime(2023, 12, 31)

data = Daily(location, start, end)
data = data.fetch()
print(data.head())
