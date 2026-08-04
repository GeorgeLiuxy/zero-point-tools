# 中国移动登录流程自动化

`scraper.py` 使用 Playwright 模拟移动端浏览器，打开上海移动相关页面，输入手机号、触发短信验证码并继续登录流程。它主要用于验证移动端页面自动化和短信登录交互，不是通用爬虫框架。

## 安装

```bash
pip install playwright
playwright install chromium
```

## 运行

```bash
cd chinamobile
python scraper.py
```

脚本会在终端提示输入手机号和收到的验证码，并以 iPhone 11 设备参数启动浏览器。

## 注意事项

- 页面 URL、元素选择器和弹窗结构可能随站点更新而失效。
- 请勿在代码、日志或截图中保存手机号、验证码、Cookie 等敏感信息。
- 短信发送和登录操作只能用于本人账号或已获授权的测试账号，并应遵守服务规则。
