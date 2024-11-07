import pandas as pd
import torch
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import nltk

# 下载并加载西班牙语停用词（仅首次运行需要）
nltk.download('stopwords')
from nltk.corpus import stopwords
spanish_stopwords = stopwords.words('spanish')

# 1. 加载评论数据
file_path = 'random_spanish_comments_with_emojis.csv'  # 替换为你的 CSV 文件路径
comments_df = pd.read_csv(file_path)

# 2. 检查设备并初始化情感分析模型
device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else -1)
sentiment_classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=device)

# 3. 进行情感分析
comments_df['Sentiment'] = comments_df['Comentario'].apply(lambda x: sentiment_classifier(x)[0]['label'])

# 4. 主题提取（使用西班牙语停用词列表）
vectorizer = CountVectorizer(max_features=15, stop_words=spanish_stopwords)
X = vectorizer.fit_transform(comments_df['Comentario'])
theme_keywords = vectorizer.get_feature_names_out()

# 5. 统计情感结果
sentiment_counts = comments_df['Sentiment'].value_counts()

# 6. 生成主题关键词的频率
word_counts = X.toarray().sum(axis=0)
topic_counts = dict(zip(theme_keywords, word_counts))

# 7. 将结果转换为 DataFrame 便于查看
sentiment_counts_df = pd.DataFrame(sentiment_counts).reset_index()
sentiment_counts_df.columns = ['Sentiment', 'Count']

topic_counts_df = pd.DataFrame(list(topic_counts.items()), columns=['Topic', 'Frequency'])

# 8. 显示情感和主题频率数据
print("情感分析分布：")
print(sentiment_counts_df)

print("\n主题关键词频率：")
print(topic_counts_df)
