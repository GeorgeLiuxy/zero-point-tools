from playwright.sync_api import sync_playwright
import time

def start_crawler(phone, code):
    with sync_playwright() as p:
        # 启动浏览器
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            **p.devices['iPhone 11']  # 模拟 iPhone 11 设备
        )

        # 创建一个新的页面
        page = context.new_page()

        # 打开登录页面
        page.goto("https://sh.ac.10086.cn/jtac/h5loginnew.jsp?source=shopnew&redirect_url=https%3A%2F%2Fwww.sh.10086.cn%2Femall%2Fstatic%2Fh5%2Fcard%2Fcard_list.html&otherNetSmart=1&loginBusType=hk")

        # 等待手机号输入框加载
        page.wait_for_selector('div.van-field__body')

        # 输入手机号
        phone_input = page.locator('div.van-field__body input.van-field__control')
        phone_input.fill(phone)

        print("手机号输入完毕，等待获取验证码...")

        # 点击获取验证码按钮
        page.click('button.van-button--primary.van-button--small')
        print("点击了获取验证码按钮，等待验证码发送...")

        # 等待验证码输入框加载
        page.wait_for_selector('input[type="text"][placeholder="请输入验证码"]')

        # 输入验证码
        code_input = page.locator('input[type="text"][placeholder="请输入验证码"]')
        code_input.fill(code)
        print("验证码输入完毕，准备提交登录...")

        # 等待登录按钮可点击
        page.wait_for_selector('button.van-button--info.van-button--normal.van-button--block.van-button--round', state='enabled')

        # 点击登录按钮
        page.click('button.van-button--info.van-button--normal.van-button--block.van-button--round')
        print("点击登录按钮，等待响应...")

        # 等待提示框弹出
        page.wait_for_selector('div.van-dialog')

        # 获取提示框中的消息
        message = page.locator('div.van-dialog__message').text_content()
        print(f"提示信息: {message}")

        # 根据提示信息判断登录是否成功
        if message == "您输入的动态密码不正确，请确认后重试。":
            print("登录失败，验证码错误，请检查输入的验证码。")
        else:
            print("登录成功！")
            # 点击确认按钮，关闭提示框
            page.click('button.van-dialog__confirm')

        # 关闭浏览器
        browser.close()

# 获取手机号和验证码
phone = input("请输入手机号: ")
code = input("请输入验证码: ")

# 调用爬虫程序逻辑（模拟浏览器操作）
start_crawler(phone, code)
