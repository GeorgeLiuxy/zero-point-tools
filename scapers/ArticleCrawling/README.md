# BBC 英语学习与文章采集实验

该目录用于采集 BBC 英语学习页面、普通 BBC 搜索文章和相关音频，并保留知乎页面适配及文件上传实验。脚本会解析列表/文章页，将正文保存为文本，并在部分流程中下载 MP3。

## 文件

- `bbc.py`：BBC Learning English 专用流程，记录已下载条目并下载文本/MP3。
- `main_bbc.py`：按搜索词抓取 BBC 文章的较完整版本。
- `main.py`、`main01.py`：通用文章列表和正文解析的迭代版本。
- `main_zhihu.py`：知乎页面结构适配实验。
- `upload.py`：阿里云 OSS 文件上传接口测试，需要额外安装 `oss2` 并配置本地凭据。
- `downloaded.txt`：已处理条目记录，用于避免重复下载。

## 安装与运行

```bash
pip install requests beautifulsoup4
cd scapers/ArticleCrawling
python bbc.py
# 或
python main_bbc.py
```

搜索词、起始 URL、请求头、输出文件名和下载目录均需在脚本中检查。输出通常保存在当前目录，包括正文文本、音频和下载记录。

## 注意事项

- 网页 DOM、分页和媒体地址变化后，解析规则需要同步更新。
- `downloaded.txt` 会影响重复任务；重新抓取前应确认是否需要清理记录。
- `upload.py` 中的 AccessKey 占位符应改为环境变量或本地密钥配置，禁止把真实云凭据写入代码。
- 设置合理的超时、重试和访问间隔，并遵守站点服务条款、robots 规则和内容版权要求。
- 文章及音频默认仍受原发布方版权约束，下载文件不因此获得再发布许可。
