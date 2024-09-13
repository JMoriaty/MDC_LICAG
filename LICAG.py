# -*- coding: utf-8 -*-
# @Time    : 2024/7/21 13:00
# @Author  : Ginger
# @FileName: LICAG.py
# @Software: PyCharm
'''
统一data和anchor都是（n*d）维度，n为样本数，d为样本维数
'''

'''
LICAG函数：
input:(X, dim, n_anchors, n_neighbors)
    X:多视图数据.多视图list(有多个视图的ndarray组成)
    dim：提取出锚点的特征维度
    n_anchor：锚点个数
    n_neighbors:计算隶属度时的k邻居个数
'''

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import random
import warnings

warnings.filterwarnings('ignore')


def LICAG(X, dim, n_anchors, n_neighbors):
    n_views = len(X)
    n_samples = X[0].shape[0]   #100

    if dim == 0:
        dim = n_anchors

    if n_neighbors == 0 or n_neighbors > n_anchors - 1:
        n_neighbors = n_anchors - 1

    X_assemble = np.hstack([x for x in X])      #数据水平拼接

    random.seed(5489)

    if n_samples > 10000:
        tmpIdx = random.sample(range(n_samples), 10000)
        subset = X_assemble[tmpIdx, :]
    else:
        subset = X_assemble

    kmeans = KMeans(n_clusters=n_anchors, max_iter=20, n_init=5, random_state=5489).fit(subset)
    anchors = kmeans.cluster_centers_

    W = csr_matrix((n_samples + n_views * n_anchors, n_samples + n_views * n_anchors))
    beg = 0
    for viewIndex in range(n_views):
        x_viewIndex = X[viewIndex]
        n_features = x_viewIndex.shape[1]
        anchor_viewIndex = anchors[:, beg:beg + n_features]

        Z = anchor_graph_construction(x_viewIndex, anchor_viewIndex, n_neighbors)

        W[n_samples + (viewIndex) * n_anchors: n_samples + (viewIndex + 1) * n_anchors, 0: n_samples] = Z
        W[0: n_samples, n_samples + (viewIndex) * n_anchors: n_samples + (viewIndex + 1) * n_anchors] = Z.T

        beg += n_features


    normalized = True
    L = laplacian(W, normalized)    #the shape of L(same of W) is (n+r*v)*(n+r*v)
    eigenvalue, eigenvactor = eigsh(L, k=dim, which='SA')
    # eigenvalue = eigenvalue[:n_points, :].T
    # matlab中返回的是一个特征值的矩阵，python中返回的是一个特征值的数组

    if eigenvalue.ndim == 1:
        eigenvalue = eigenvalue[:, np.newaxis]
    eigenvalue = eigenvalue[:n_samples, :].T
    return eigenvalue, eigenvactor[:n_samples, :]


def anchor_graph_construction(data, anchors, m):
    n = data.shape[0]
    n_anchors = anchors.shape[0]

    E = cdist(anchors, data, 'sqeuclidean')

    SortedE = np.sort(E, axis=0)
    Eim1 = np.tile(SortedE[m, :], (n_anchors, 1))

    IndMask = np.zeros((n_anchors, n))
    Ind = np.argsort(E, axis=0)
    for i in range(n):
        IndMask[Ind[:m, i], i] = 1

    E_numerator = E * IndMask
    Eim1_numerator = Eim1 * IndMask

    denominator = m * Eim1 - np.sum(E_numerator, axis=0) + np.finfo(float).eps
    Z = (Eim1_numerator - E_numerator) / denominator

    return csr_matrix(Z)


def laplacian(W, normalized):
    n = W.shape[0]
    d = np.array(W.sum(axis=1)).flatten()
    if normalized:
        Dn = csr_matrix(np.diag(d ** -0.5))
        L = eye(n) - Dn @ W @ Dn
    else:
        D = csr_matrix(np.diag(d))
        L = D - W
    return L


if __name__ == '__main__':
    np.random.seed(100)
    X = [np.random.rand(100, 5), np.random.rand(100, 10)]
    dim = 6
    n_anchors = 10
    n_neighbors = 5

    print("the type of x",type(X))
    print("the type of x[0]",type(X[0]))

    eigenvalue, eigenvactor = LICAG(X, dim, n_anchors, n_neighbors)

    print("the eigenvalue",eigenvalue)
    print("the  eigenvactor",eigenvactor)


    from time import time
    start =time()
    t0 = time()
    t1 = time()
    print("Total time:",(t1 - t0))

