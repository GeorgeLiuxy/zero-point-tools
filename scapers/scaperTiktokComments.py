import requests,json

post_url = 'https://www.tiktok.com/@medalgamexclawcrane/video/7427022544600960263'

post_id = post_url.split('/')

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36',
    'Content-Type': 'application/json'
    
}

def req():

    url = 'https://www.tiktok.com/api/comment/list/?WebIdLastTime=1726717185&aid=1988&app_language=ja-JP&app_name=tiktok_web&aweme_id=7427022544600960263&browser_language=zh-CN&browser_name=Mozilla&browser_online=true&browser_platform=MacIntel&browser_version=5.0%20%28Macintosh%3B%20Intel%20Mac%20OS%20X%2010_15_7%29%20AppleWebKit%2F537.36%20%28KHTML%2C%20like%20Gecko%29%20Chrome%2F130.0.0.0%20Safari%2F537.36&channel=tiktok_web&cookie_enabled=true&count=20&current_region=JP&cursor=20&data_collection_enabled=true&device_id=7416193801533195783&device_platform=web_pc&enter_from=tiktok_web&focus_state=false&fromWeb=1&from_page=video&history_len=5&is_fullscreen=false&is_non_personalized=false&is_page_visible=true&odinId=6940294814341301253&os=mac&priority_region=US&referer=&region=JP&screen_height=1050&screen_width=1680&tz_name=Asia%2FShanghai&user_is_login=true&verifyFp=verify_m35onwem_rRfbZXis_iyx1_49cY_8jbL_mHacdHiV2YBp&webcast_language=zh-Hans&msToken=wZKqNTlStfI3oYl2qN4YO75egMeP0455CQ7ZdVp4D7unkCAN5J6CTbiF86fC6pSt604m4lIa98b6z6S6ffRzcVHhVLkjoQSJLTtznlDbtwNrDKdbnJ1B-YYwuKsokQ6BvnJ1JmXENzhS08M0DRDt4kLO&X-Bogus=DFSzswVLgJiANrZJtseGhDLNKBPC&_signature=_02B4Z6wo00001o3dAkwAAIDB3yIiVEPlTEKN3QbAAMRo5b'
    response = requests.get(url=url, hdaders=headers)
    info = response.text
    raw_data = json.loads(info)
    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=4)

def parser():
    pass


print(post_id)