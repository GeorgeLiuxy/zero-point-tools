# 步骤：
# 通过交叉验证选择最优的模型和超参数。
# 使用网格搜索进行超参数优化。

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from AI.study.step1 import X_train, y_train

# 定义参数网格
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)

# 训练并选择最优模型
grid_search.fit(X_train, y_train)

# 输出最优参数和模型
print("Best Parameters:", grid_search.best_params_)
