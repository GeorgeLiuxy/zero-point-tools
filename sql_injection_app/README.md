# MobileBERT SQL 注入文本检测应用

一个 Flask Web 原型，使用 Hugging Face 上的 MobileBERT 序列分类模型判断输入文本更接近正常 SQL 还是 SQL 注入样本，并返回分类结果和模型置信度。

## 结构

```text
sql_injection_app/
├── app.py                    # Flask 入口，监听 5001 端口
├── model_loader.py           # 模型下载、缓存和 CPU 推理
├── saved_model/mobilebert_sql/  # 本地模型缓存
└── templates/index.html      # Web 页面
```

## 安装与启动

本目录当前没有 `requirements.txt`，可直接安装代码所需依赖：

```bash
pip install flask torch transformers
cd sql_injection_app
python app.py
```

浏览器打开 `http://localhost:5001`。应用首次启动时优先读取 `saved_model/mobilebert_sql`；若本地目录不可用，会下载 `cssupport/mobilebert-sql-injection-detect` 并缓存到该目录。

## 推理流程

1. 对输入 SQL 文本进行截断和分词，最大长度为 256。
2. 在 CPU 上执行二分类推理。
3. 按最高 softmax 概率返回“正常语句”或“注入攻击”以及置信度。

## 限制与部署提示

- 这是统计模型判断，不能替代参数化查询、最小权限、输入验证、WAF 和安全审计。
- 当前标签含义按代码假设类别 `1` 为注入；更换模型时必须核对 `id2label`，不能只沿用数值映射。
- `app.py` 开启 `debug=True` 并绑定 `0.0.0.0`，只适合本地开发。对外部署前应关闭调试、使用正式 WSGI 服务并增加访问控制。
- 不要把生产数据库语句、密钥或敏感数据提交给未经审核的服务。
