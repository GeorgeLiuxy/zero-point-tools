import os
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple

# ==================== 环境配置 ====================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用 HuggingFace 镜像

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SQLiDetector")

# ==================== 模型路径设置 ====================
MODEL_ID = "cssupport/mobilebert-sql-injection-detect"
MODEL_PATH = "./saved_model/mobilebert_sql"

# ==================== 模型加载函数 ====================
def load_model_and_tokenizer():
    try:
        if os.path.exists(MODEL_PATH) and os.path.isdir(MODEL_PATH):
            logger.info("🔍 正在从本地加载模型...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        else:
            logger.info(f"🌐 本地模型不存在，正在从 HuggingFace 镜像下载并保存到：{MODEL_PATH}")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
            os.makedirs(MODEL_PATH, exist_ok=True)
            tokenizer.save_pretrained(MODEL_PATH)
            model.save_pretrained(MODEL_PATH)

        model = model.to("cpu")
        model.eval()
        logger.info("✅ 模型加载成功，已切换至 CPU 模式")
        return tokenizer, model

    except Exception as e:
        logger.error(f"❌ 模型加载失败：{e}")
        raise RuntimeError("模型加载失败，请检查网络连接或模型路径。")

# ✅ 初始化模型（仅加载一次）
tokenizer, model = load_model_and_tokenizer()

# ==================== 检测函数 ====================
def predict_sql_injection(sql_text: str) -> Tuple[str, float]:
    """
    使用 MobileBERT 模型检测 SQL 语句是否存在注入攻击
    返回: label（正常语句 / 注入攻击）, 置信度百分比
    """
    try:
        if not sql_text or not isinstance(sql_text, str):
            raise ValueError("SQL输入为空或格式不正确")

        inputs = tokenizer(sql_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = round(probs[0][predicted_class].item() * 100, 2)
        label = "注入攻击" if predicted_class == 1 else "正常语句"

        logger.info(f"检测完成 | 结果: {label} | 置信度: {confidence}%")
        return label, confidence

    except Exception as e:
        logger.error(f"⚠️ 检测过程中出错：{e}")
        return "检测失败", 0.0
