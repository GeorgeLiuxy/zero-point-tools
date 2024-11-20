# 数据集：使用经典的"客户数据"集，通过K-means聚类算法进行市场细分。
# 步骤：
# 导入数据并进行预处理（如去除缺失值、标准化等）。
# 使用K-means进行聚类，并选择合适的K值。
# 评估聚类效果（如轮廓系数、肘部法则等）。
# 导入库
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 1. 生成示例数据
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# 2. K-means聚类
kmeans = KMeans(n_clusters=4, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# 3. 可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
plt.title("K-means Clustering")
plt.show()
