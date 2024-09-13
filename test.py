import numpy as np
from scipy.spatial.distance import cdist

# 创建两个示例矩阵 A 和 B
A = np.random.rand(5, 3)  # 5 行 3 列矩阵
B = np.random.rand(4, 3)  # 4 行 3 列矩阵

# 计算欧氏距离
D = cdist(A, B, metric='euclidean')

print(D)
