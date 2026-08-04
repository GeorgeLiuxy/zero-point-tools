# 在线课程学习自动化实验

该目录保存多个面向在线培训平台的浏览器/API 自动化脚本，包含登录界面、课程和章节读取、视频播放/计时、学习进度更新以及部分验证码识别。不同文件对应不同平台或不同实现阶段，没有统一接口。

## 主要入口

| 文件 | 说明 |
| --- | --- |
| `main.py` | Tkinter + Selenium 图形界面版本，驱动 Chrome 完成登录和课程播放 |
| `new_auto.py`、`new_auto_ui.py` | 基于 HTTP API 的新版流程及 GUI |
| `cme28_91huayi_com.py` | 91huayi/相关培训平台流程，含 Tesseract 验证码识别 |
| `nmgcf_chinahrt_com.py` | chinahrt 平台的课程与视频进度处理 |
| `zjpx_nmgdbrc_com.py` | 另一培训平台适配脚本 |
| `play_video.py`、`toutiao.py`、`test.py` | 播放、调试和阶段性实验 |
| `encrypt.js`、`sm_encrypt.js` | 登录或请求参数加密辅助脚本 |

## 环境

优先安装目录内依赖：

```bash
cd auto_video_study
pip install -r requirements.txt
```

当前 `requirements.txt` 中包含 `tk` 和含义不明确的 `k` 条目，直接安装可能失败；Tkinter 通常随桌面版 Python 提供，`k` 应先核实或移除。部分版本还需要 Chrome、Node.js、Tesseract OCR 和与浏览器匹配的驱动。Selenium 版本通常由 `webdriver_manager` 管理驱动。

## 运行

```bash
python main.py
# 或按目标平台运行对应脚本
python new_auto_ui.py
```

## 配置与安全

- 平台 URL、训练计划 ID、接口字段和页面选择器多为脚本内配置，运行前必须核对。
- 不要把真实账号、密码、Cookie、Token、私钥或登录响应提交到仓库；建议改用环境变量或本地忽略的配置文件。
- 目录中已有 `.pem`、`.der` 文件；若它们曾用于真实环境，应视为已暴露凭据并立即轮换，仓库只保留无效测试密钥。
- 平台接口和页面结构变化会导致脚本失效。
- 仅应在获得授权并符合平台规则的情况下使用，学习进度应真实反映本人完成情况。
