# app.py
import os
import logging
from flask import Flask, request, render_template
from model_loader import predict_sql_injection

# ==================== 基础配置 ====================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024  # 限制请求体大小（10KB以内）
app.config['JSON_AS_ASCII'] = False           # 防止中文乱码
app.config['SECRET_KEY'] = os.urandom(24)     # CSRF等场景使用

# 设置日志（控制台输出，可拓展为写文件）
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SQLiApp")

# ==================== 路由定义 ====================
@app.route("/", methods=["GET", "POST"])
def index():
    result, confidence = None, None
    sql_input = ""

    if request.method == "POST":
        try:
            sql_input = request.form.get("sql_input", "").strip()
            if sql_input:
                result, confidence = predict_sql_injection(sql_input)
            else:
                result, confidence = "输入为空", 0.0
        except Exception as e:
            logger.error(f"处理请求出错：{e}")
            result, confidence = "检测失败", 0.0

    return render_template("index.html", result=result, confidence=confidence, sql_input=sql_input)

# ==================== 应用入口 ====================
def run_app():
    try:
        logger.info("🚀 SQL注入检测系统启动中...")
        app.run(
            host="0.0.0.0",      # 可远程访问
            port=5001,
            debug=True,         # 正式部署可改为 False
            use_reloader=False  # 防止重启时重复加载
        )
    except Exception as e:
        logger.exception(f"❌ Flask 启动失败：{e}")

if __name__ == "__main__":
    run_app()
