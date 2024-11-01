# please use Python 3.8.* 版本
import itchat
import html
import re

# 定义要监听的群聊名称
TARGET_GROUP_NAME = "测试"

# 更新后的正则表达式模式，适配多行文本
MESSAGE_PATTERN = r"(\d{6}-\d{4}).*?XX.*?\+ (\w+) XX"

@itchat.msg_register(itchat.content.TEXT, isGroupChat=True)
def group_text_reply(msg):
    # 检查消息是否来自目标群聊
    if msg['User']['NickName'] == TARGET_GROUP_NAME:
        # 匹配消息内容
        match_result = message_match(msg.text)

        # 检查是否找到匹配项
        if match_result:
            order_id, account_id = match_result
            print(f">>>>>>>>Match found! Order ID: {order_id}, Account Name: {account_id}")

            # 构造发送的消息内容
            order_message = f"JDD {order_id}"

            # 查找用户并发送消息（实际发送逻辑已注释）
            user = get_user_by_id(account_id)
            if user:
                itchat.send(order_message, toUserName=user[0]['UserName'])
                print(f"Message sent to {account_id}: {order_message}")
            else:
                print(f"User with ID '{account_id}' not found.")
        else:
            print("No match found for the specified pattern.")


def message_match(msg_text):
    # 解码消息内容
    msg_text = html.unescape(msg_text)
    # 将所有换行符替换为空格
    msg_text = msg_text.replace("\n", " ")
    print("Testing message:", msg_text)  # 输出待测试的消息内容
    # 匹配特定格式的消息内容
    match = re.search(MESSAGE_PATTERN, msg_text)
    if match:
        order_id = match.group(1).strip()  # 提取动态编号
        account_name = match.group(2).strip()  # 提取JDD人账号
        print(f"Match found! Order ID: {order_id}, Account Name: {account_name}")
        return order_id, account_name
    else:
        print("No match found for the specified pattern.")
        return None


def get_user_by_id(account_id):
    # 通过微信ID查找用户
    if account_id == 'XXX':
        return get_user_by_nike('ZXXX')


def get_user_by_nike(nike_name):
    # 查找用户并发送消息
    user = itchat.search_friends(nickName=nike_name)
    return user


# 登录并开始运行
itchat.auto_login()
itchat.run()