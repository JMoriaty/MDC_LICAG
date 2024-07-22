import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

# 创建一个稀疏对角矩阵
N = 100
k = 6
diagonals = [np.random.rand(N)]
A = diags(diagonals, [0])

# 计算前k个最大的特征值和对应的特征向量
w, v = eigsh(A, k=k, which='LM')

print("特征值shape:", w.shape)
print("特征向量shape:", v.shape)
