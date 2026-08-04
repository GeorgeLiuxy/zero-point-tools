# 安居客商业地产 Selenium 脚本集合

该目录保存安居客上海站商业地产信息采集的多个版本，包括普通版、快速版、带断点进度版以及历史/分阶段脚本。主要提取联系电话、经纪人姓名和虚拟号码标记。

## 推荐入口

| 文件 | 说明 |
| --- | --- |
| `sp_cz_flag_scaper.py` | 商铺出租，断点续跑 |
| `xzl_cz_flag_scraper.py` | 写字楼出租，断点续跑 |
| `xzl_sale_flag_scraper.py` | 写字楼出售，断点续跑 |
| `fast_ajk_scraper.py` | 写字楼出租快速实验版 |
| `fast_ajk_xzl_sale_scraper.py` | 写字楼出售快速实验版 |
| `ajk_sp_cz_scraper.py` | 商铺出租普通版 |
| `ajk_xzl_cz_scraper.py` | 写字楼出租普通版 |
| `ajk_xzl_sale_scraper.py` | 写字楼出售普通版 |
| `old/`、`script/` | 历史版本和基于链接文件的阶段性脚本 |

断点版会生成 `completed_pages*.txt` 和业务 CSV；普通/快速版使用各自硬编码的 CSV 文件名和起始页码。

## 安装与运行

```bash
cd webscraper
pip install -r requirements.txt
python xzl_cz_flag_scraper.py
```

依赖为 Selenium、webdriver-manager、fake-useragent，且本机需要 Chrome。

## 运行前检查

1. 将页码范围改成少量测试页。
2. 核对上海站 URL、列表/详情页选择器和超时时间。
3. 核对输出 CSV 是覆盖模式还是追加模式。
4. 调试时关闭 headless，确认点击和窗口切换行为。
5. 不要同时运行会写入同一 CSV 或进度文件的脚本。

## 使用边界

这些脚本依赖页面结构且没有稳定 API 保证。请遵守网站服务条款、robots 规则、访问频率限制和个人信息保护要求；不要绕过访问控制，也不要把采集到的手机号等敏感数据公开提交到仓库。
