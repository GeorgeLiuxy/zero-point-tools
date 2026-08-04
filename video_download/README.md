# 多平台视频下载实验

该目录保存 Bilibili、抖音、快手、头条和西瓜视频等平台的下载或页面解析脚本，并混有一个在线课程平台自动化版本。各脚本是独立实验，不是统一下载器。

## 文件

| 文件 | 说明 |
| --- | --- |
| `B_site.py` | Bilibili 页面/媒体下载实验 |
| `douyin.py` | 抖音链接解析与下载 |
| `kuaishou.py` | 快手链接解析与下载 |
| `toutiao.py` | Selenium 方式处理头条页面 |
| `xigua.py` | 西瓜视频 PyQt 图形界面实验 |
| `nmgcf_chinahrt_com.py` | 在线学习平台脚本，与通用视频下载无直接关系 |

## 环境

```bash
cd video_download
pip install -r requirements.txt
pip install yt-dlp selenium webdriver-manager
```

部分脚本依赖系统中的 FFmpeg；请单独安装并确保 `ffmpeg` 可从命令行访问。GUI 版本还需要 PyQt。

## 使用

先在目标脚本中替换示例 URL、输出目录和浏览器配置，再运行：

```bash
python B_site.py
# 或
python xigua.py
```

## 注意事项

- 平台页面、接口签名和反自动化策略变化后，脚本可能需要更新。
- 只能下载本人拥有权利、已获授权或平台明确允许离线保存的内容，并遵守版权和服务条款。
- Cookie、Token、账号信息和下载记录不应提交到仓库。
