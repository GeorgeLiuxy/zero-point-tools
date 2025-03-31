import requests

url = "https://www.zhihu.com/api/v4/search_v3?gk_version=gz-gaokao&t=general&q=%E6%B3%95%E5%BE%8B&correction=1&offset=0&limit=20&filter_fields=&lc_idx=0&show_all_topics=0&search_source=Normal"

payload={}
headers = {
    'pragma': 'no-cache',
    'cache-control': 'no-cache',
    'sec-ch-ua-platform': '"macOS"',
    'x-app-za': 'OS=Web',
    'sec-ch-ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    'x-api-version': '3.0.91',
    'x-zse-96': '2.0_h=FRIJquxw4RuPOXomDM7QTgI+Z2kJsZ=bAuk7L8F90brX7V7GFO1e+MkX/iY6x=',
    'sec-ch-ua-mobile': '?0',
    'x-requested-with': 'fetch',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'accept': '*/*',
    'sec-fetch-mode': 'cors',
    'sec-fetch-dest': 'empty',
    'accept-encoding': 'gzip, deflate, br, zstd',
    'accept-language': 'zh-CN,zh;q=0.9',
    'cookie': '_xsrf=RR3OE6eiigVc5Dzb5gZRQZZfMLUYnaiN; _zap=4330549b-701d-40c6-997e-45140e278dec; d_c0=AQDS_1ClOxmPTirV27MXRzjUO8bvOxRHPBI=|1726279489; q_c1=8bf1abfccc084042940428774274eb2e|1729241923000|1729241923000; Hm_lvt_98beee57fd2ef70ccdd5ca52b9740c49=1728457369,1729043835,1730632757; z_c0=2|1:0|10:1737095670|4:z_c0|80:MS4xMDFBSU53QUFBQUFtQUFBQVlBSlZUZlpIZDJncWptdnF2cl9MSXBvcVdlM003bVVHZld3LUJBPT0=|d3bbde02528480d3448d42e3e9ce5aec0f60b7c38a84b9de44906b4cf9d125ad; __zse_ck=004_/3pe5WIq1LNFgSvtc4XwnqXm2IuVhlFkfo=ZHjxrbYhpQsY31qxfGiT7M8O6oJGNcWqaMR9jSnRIkQrG4M8JADUtLGLdzCdB17I8ixEnraYTZ1AjCY8d=2A=xjBEskts-54AZyIcfmNMMAG/b6UBm8hCJf9ywfTgJExhGXcrmqvFvm0naCp/XWay+sKt6OwTkLhT2fRl/WaKXbIbn5fVEkbRe2jh/f1T2vVHzo+Nw/h2xR3dSsykETobEfpPk3kbl; tst=r; SESSIONID=23jpMooTdPznFObk7SQXE1iJdxmglxj265FupnKCFUd; JOID=V1EUA0oH1hyCEQoGJgIuiSBUFlI7Oe1381hpYXNnim7tVkFoRb-Cl-AZDg8mtNyrzxapgng-e1uS6WMRH2EfJ5s=; osd=Wl0UA0wK2hyCFwcKJgIohCxUFlQ2Ne139VVlYXNhh2LtVkdlSb-Cke0VDg8gudCrzxCkjng-fVae6WMXEm0fJ50=; edu_user_uuid=edu-v1|5bffd8a2-55c0-49f9-b0a6-32d0fee54b45; BEC=b7b0f394f3fd074c6bdd2ebbdd598b4e; priority=u=1, i',
}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)
