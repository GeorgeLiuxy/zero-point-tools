# 基于深度学习的 SQL 注入检测系统

本项目是一个基于 Python Flask 和 HuggingFace Transformers 构建的 **SQL 注入检测系统**，通过加载训练好的 MobileBERT 模型来识别 SQL 查询语句中是否存在注入攻击行为。系统支持 Web 表单交互，用户可输入 SQL 语句并实时获取检测结果与风险分析。

---

## 📦 项目结构

```plaintext
sql_injection_app/
├── app.py                  # Flask 应用主入口（带日志与异常处理）
├── model_loader.py         # 模型加载与推理逻辑（使用 MobileBERT）
├── templates/
│   └── index.html          # 前端交互界面（HTML5）
├── requirements.txt        # 依赖环境
└── README.md               # 项目说明文档（即本文件）
```

---

## 🚀 快速启动

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 启动应用：

```bash
python app.py
```

3. 访问系统：

浏览器打开 [http://localhost:5000](http://localhost:5000)，输入 SQL 语句进行检测。

---

## 🔍 检测说明

模型使用 `cssupport/mobilebert-sql-injection-detect`，为微调后的轻量化 BERT 模型，具备以下特点：

- 对输入 SQL 进行二分类：`正常语句` 或 `注入攻击`
- 返回判断结果 + 置信度（百分比）
- 支持本地 CPU 推理，适配 macOS / Windows / Linux 多平台

---

## 🧪 测试样例

以下 SQL 可用于验证系统是否正常运行及模型效果：

### ✅ 安全语句（应识别为正常）

```sql
SELECT * FROM users WHERE username = 'alice' AND password = 'mypassword'
```

### ❌ 注入攻击语句（应识别为注入攻击）

```sql
SELECT * FROM users WHERE username = 'admin' OR '1'='1' -- 
```

---

## 💡 技术栈

- Python 3.8+
- Flask 2.2+
- PyTorch 2.1+
- Transformers 4.36+
- HTML5 + CSS（模板渲染）

---

## 🧰 功能亮点

- ✅ 多平台兼容（Mac / Windows / Linux）
- ✅ 禁用 tokenizer 并发，防止资源泄漏（Mac 安全支持）
- ✅ 模型与前端完全解耦，便于二次开发
- ✅ 可扩展风险解释模块（匹配注入类型）

---

## 📄 License

MIT License © 2025 YourName