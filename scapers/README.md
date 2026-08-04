# 数据采集与内容分析实验

该目录保存新闻、文章、短视频评论、图片和招聘信息等多个独立采集/分析脚本。目录名 `scapers` 为历史命名；各脚本没有统一配置、依赖或入口。

## 内容导航

| 路径 | 说明 | 状态 |
| --- | --- | --- |
| `ArticleCrawling/` | BBC 英语学习文章、音频及知乎等内容采集 | 可实验，需检查页面结构 |
| `googleNewsScraper.py` | Google News 搜索/新闻抓取 | 实验脚本 |
| `google_image_anasys.py` | Google 图片采集、分类和 YOLO 分析组合实验 | 需模型与浏览器环境 |
| `scaperDouyinComments.py` | 抖音评论采集 | 平台接口敏感，需更新配置 |
| `scaperTiktokComments.py` | TikTok 评论采集 | 平台接口敏感，需更新配置 |
| `job_scraper/` | Scrapy 招聘信息项目 | Scrapy 工程 |
| `example_spider/` | Scrapy 示例工程 | 含历史构建目录 |
| `medium_main.py` | 预留脚本 | 当前为空 |

## 环境与运行

依赖因脚本而异，常见包包括 `requests`、`beautifulsoup4`、`selenium`、`scrapy`、`pandas`、`opencv-python`、`torch`、`torchvision` 和 YOLO 相关库。先阅读目标脚本，再单独运行：

```bash
cd scapers
python googleNewsScraper.py
```

Scrapy 项目应进入包含 `scrapy.cfg` 的工程目录后使用 `scrapy crawl <spider>`。

## 使用边界

- URL、Cookie、请求头、关键词和输出路径多为脚本内配置。
- 页面结构、反自动化机制和接口签名可能随时变化。
- 运行前应确认站点服务条款、robots 规则、版权和个人信息处理要求，并设置合理的访问频率。
- 不要提交真实 Cookie、Token、手机号或账号信息。
