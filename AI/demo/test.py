from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy

# 加载情感分析模型
sentiment_classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# 示例评论数据
comments = [
    "Me encanta la calidad de la cámara de este teléfono.",
    "La duración de la batería podría ser mejor.",
    "Resolución de pantalla increíble y excelente diseño.",
    "No estoy satisfecho con la última actualización del software, es demasiado lenta."
]

# 情感分析
sentiments = [sentiment_classifier(comment)[0] for comment in comments]
print("情感分析结果:", sentiments)

# 主题建模
vectorizer = CountVectorizer(stop_words="english", max_features=1000)
comment_matrix = vectorizer.fit_transform(comments)
lda = LatentDirichletAllocation(n_components=2, random_state=0)
lda.fit(comment_matrix)

# 提取主题关键词
topic_keywords = []
for i, topic in enumerate(lda.components_):
    topic_keywords.append([vectorizer.get_feature_names_out()[j] for j in topic.argsort()[-5:]])
print("主题建模关键词:", topic_keywords)

# 生成用户偏好标签
user_preferences = {}
for idx, comment in enumerate(comments):
    sentiment = sentiments[idx]["label"]
    score = sentiments[idx]["score"]
    if sentiment in ["4 stars", "5 stars"] and score > 0.7:
        user_preferences[f"Comentario {idx+1}"] = "Positivo"
    elif sentiment in ["1 star", "2 stars"]:
        user_preferences[f"Comentario {idx+1}"] = "Negativo"
    else:
        user_preferences[f"Comentario {idx+1}"] = "Neutro"

print("用户偏好标签:", user_preferences)
