# 安居客商业地产联系电话采集

该目录使用 Selenium 访问安居客上海商业地产列表和详情页，提取联系电话、经纪人姓名及虚拟号码标记。三个入口分别处理商铺出租、写字楼出租和写字楼出售，并用文本文件记录已完成页码以支持断点续跑。

## 脚本

| 入口 | 业务 | 默认输出 | 进度文件 |
| --- | --- | --- | --- |
| `sp_cz_flag_scaper.py` | 商铺出租 | `商铺出租13.csv` | `completed_pages.txt` |
| `xzl_cz_flag_scraper.py` | 写字楼出租 | `写字楼租赁11.csv` | `completed_pages_xzl_zu.txt` |
| `xzl_sale_flag_scraper.py` | 写字楼出售 | `写字楼销售.csv` | `completed_pages_xzl_shou.txt` |

CSV 字段为 `电话号码`、`经纪人姓名`、`是否虚拟号码`。

## 安装与运行

```bash
cd anjuke_phone_scraper
pip install -r requirements.txt
python xzl_cz_flag_scraper.py
```

依赖包括 Selenium、webdriver-manager 和 fake-useragent。脚本会自动管理 ChromeDriver，但本机仍需安装兼容的 Chrome。

## 配置

页码上限、上海站 URL、等待时间、无头模式、资源拦截和输出文件名均直接写在脚本中。首次运行前建议先缩小页码范围并显示浏览器，确认页面选择器仍然有效。

## 注意事项

- 脚本在模块加载后即开始执行，没有命令行参数和统一配置文件。
- 页面结构、验证码和反自动化策略变化后可能无法工作。
- 删除对应的 `completed_pages*.txt` 才会从第一页重新开始，但这样会造成重复采集，应先备份或清理 CSV。
- 联系电话和经纪人信息涉及个人信息，只能在具备合法依据、获得必要授权并符合网站规则时采集和使用；请控制请求频率。
