# 微信消息与界面自动化实验

该目录主要基于 `itchat` 和 Tkinter 实现微信登录、群消息监听、正则匹配、自动回复、联系人统计及图形界面控制，并包含一个安居客微信接口请求实验。

## 文件

- `wechat_auto_reply.py`：监听指定群聊，匹配订单/账号格式并向联系人发送消息。
- `wechat_auto_ui.py`：带二维码和日志窗口的群消息监听界面。
- `wechat_auto_ui_jd.py`：面向另一消息格式的 GUI 版本。
- `statistics_wechat_infos.py`：登录后记录或统计收到的文本/分享消息。
- `scraper_ajk_zj_mobiles.py`：带固定请求头的安居客微信接口测试。
- `xxx.py`：阶段性服务/消息处理辅助代码。

## 安装与运行

```bash
pip install itchat pillow requests
cd wechat
python wechat_auto_ui.py
```

Tkinter 通常随桌面版 Python 安装。命令行版本可运行 `python wechat_auto_reply.py`。

## 配置

运行前应修改目标群名称、消息正则、特殊账号映射和回复模板。建议把这些值移入本地配置文件，不要把真实微信 ID、群名、Cookie、签名或联系人信息提交到仓库。

## 限制与安全

- `itchat` 依赖非官方网页版接口，登录能力和稳定性可能随平台策略变化。
- 自动发送消息可能造成误发或频率限制，应先在测试群中验证并加入人工确认、去重和速率限制。
- 联系人、群消息和手机号属于敏感数据，必须获得授权并遵守隐私与平台规则。
- `scraper_ajk_zj_mobiles.py` 中的请求签名和会话字段可能已失效，使用前应清理并改为本地配置。
