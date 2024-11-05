# 安居客商业地产信息爬取工具

本项目包含多个脚本，用于从安居客平台爬取商业地产（租赁和销售）信息。它可以提取每个房源的电话号码、经纪人姓名以及是否为虚拟号码等信息。该工具使用 Selenium 进行自动化浏览器操作。

## 功能特点

- **断点续爬**：每个脚本会记录上次成功爬取的页面，若爬取过程被中断，可以从上次记录的页码继续执行。
- **无头模式**：默认在无头模式下运行，以提高效率。
- **数据持久化**：提取的数据将被保存到 CSV 文件中，包含电话号码、经纪人姓名和是否为虚拟号码的信息。
- **随机延迟**：在页面请求之间添加随机延迟，降低被检测的风险。
- **选择性加载资源**：禁用图片、视频和样式表的加载以优化性能。

## 先决条件

- 安装 **Python 3.x**
- 安装 **Chrome 浏览器**（确保与 `webdriver_manager` 使用的版本兼容）
- **Chrome Driver** 由 `webdriver_manager` 自动管理

## 安装

1. 克隆此仓库：（暂未上传至GIT）
```bash
   git clone https://github.com/your-username/anjuke-webscraper.git
   cd anjuke-webscraper
```


## 在项目文件夹下运行以下命令以安装所需依赖：
```bash
    pip install -r requirements.txt
```

## 使用方法

每个脚本对应一个具体的爬取任务。包括以下脚本：

xzl_rental_scraper.py - 从安居客的商业写字楼租赁板块爬取房源信息。
xzl_sale_scraper.py - 从安居客的商业写字楼销售板块爬取房源信息。
sp_rental_scraper.py - 从安居客特定的商业租赁类别中爬取房源信息。
脚本会记录每个已完成的页面到文件中（例如：completed_pages_xzl_zu.txt、completed_pages_xzl_shou.txt等），在下次启动时从记录的页码继续。

运行脚本
可以直接用 Python 运行任一脚本：
```bash
python xzl_rental_scraper.py
```
脚本将自动在无头模式下打开 Chrome 浏览器，逐页访问页面并提取所需数据。结果会保存到项目目录下的 CSV 文件中（例如，写字楼租赁.csv）。

输出文件

每个脚本会生成一个 CSV 文件，包含以下格式的数据：

电话号码	       经纪人姓名	   是否虚拟号码
123-4567-8901	张三      	是
098-7654-3210	李四      	否

日志记录

每个脚本会输出运行时信息到控制台，便于追踪爬取进度，包括：

   当前页码
   提取的链接
   成功或跳过的条目
   发生的错误
   异常处理

脚本在被中断时会记录最后完成的页码。若发生中断，重新启动脚本时将从上次记录的页码继续爬取，避免重复工作。

自定义设置

无头模式：若要禁用无头模式，可以注释掉或删除 Chrome 选项设置中的 --headless 参数。
User-Agent 随机化：每个脚本在每次启动时使用随机 User-Agent，以减少被检测的风险。可以在 UserAgent 设置中自定义此行为。
依赖包

以下 Python 包是必需的（在 requirements.txt 文件中提供）：

selenium
webdriver_manager
fake_useragent
许可证

本项目使用 MIT 许可证。
