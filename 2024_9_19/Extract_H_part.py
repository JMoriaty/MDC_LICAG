# -*- coding: utf-8 -*-
# @Time    : 2024/7/21 13:00
# @Author  : Ginger
# @FileName: Extract_H.py
# @Software: PyCharm
'''
统一data和anchor都是（n*d）维度，n为样本数，d为样本维数
'''
from torch.utils.data import DataLoader

'''
Extract_H函数：
input:(X, dim, n_anchors, n_neighbors)
    X:多视图数据.多视图list(有多个视图的ndarray组成)
    dim：提取出锚点的特征维度
    n_anchor：锚点个数
    n_neighbors:计算隶属度时的k邻居个数
'''

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist
import random
from  sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

from config import get_config
import torch
from torch.utils.data import DataLoader
from utils import multiViewDataset,cluster_acc,multiViewDataset2,log_header
import skfuzzy as fuzzy
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import warnings
warnings.filterwarnings('ignore')


def Extract_H(X, dim, n_anchors, n_neighbors):
    n_views = len(X)
    n_samples = X[0].shape[0]   #100

    if dim == 0:
        dim = n_anchors

    if n_neighbors == 0 or n_neighbors > n_anchors - 1:
        n_neighbors = n_anchors - 1

    X_assemble = np.hstack([x for x in X])      #数据水平拼接 n_sample * n_feature

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
    eigenvalue, eigenvector = eigsh(L, k=dim, which='SA')
    # eigenvalue = eigenvalue[:n_points, :].T
    # matlab中返回的是一个特征值的矩阵，python中返回的是一个特征值的数组

    if eigenvalue.ndim == 1:
        eigenvalue = eigenvalue[:, np.newaxis]
    eigenvalue = eigenvalue[:n_samples, :].T
    return eigenvalue, eigenvector[:n_samples, :]


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
    args = get_config()

    if args.dataset=='HW':
        dataset = multiViewDataset2(args.dataset, args.viewNumber, True)
    else:
        dataset = multiViewDataset(args.dataset, args.viewNumber, True)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    for batch_index, (x, y, _) in enumerate(dataloader):
        labels = y
        X = x

    eigenvalue, eigenvector = Extract_H(X, args.dimofH,args.n_anchors,args.n_neighbors)


    # print("the eigenvalue",eigenvalue)
    print("the the eigenvector matrix:\n",eigenvector)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)
    kmeans.fit_predict(eigenvector)
    y_pred_temp = kmeans.labels_

    acc_kmeans = cluster_acc(labels, y_pred_temp)
    nmi_kmeans = nmi_score(labels, y_pred_temp)
    ari_kmeans = ari_score(labels, y_pred_temp)

    print(f'Dataset:{args.dataset}')
    print(f'KMEANS(H):\t\tacc:{acc_kmeans:f}, nmi:{nmi_kmeans:.4f}, ari:{ari_kmeans:.4f}')

    # 初始化融合矩阵F
    _, U_temp, _, _, _, _, _ = fuzzy.cluster.cmeans(
        eigenvector.T,  # 注意：数据需要是转置的格式，形状为 n_features x n_samples
        c=args.n_clusters,  # 簇的数量
        m=2,  # 模糊参数（通常设为 2）
        error=0.005,  # 终止条件的误差
        maxiter=1000,  # 最大迭代次数
        init=None,  # 初始隶属度矩阵（可以为空）
        seed=256  # 随机种子
    )
    # U = torch.Tensor(U_temp.T)
    U = U_temp.T
    label_FCM = U.argmax(1)
    acc_FCM = cluster_acc(labels, label_FCM)
    nmi_FCM = nmi_score(labels, label_FCM)
    ari_FCM = ari_score(labels, label_FCM)

    print(f'FCM(H):\t\tacc:{acc_FCM:f}, nmi:{nmi_FCM:.4f}, ari:{ari_FCM:.4f}')



    print(f'FCM on H, the U matrix is:\n{U}')


    print("---------------This is std_data process--------------")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(eigenvector)

    # 或者进行归一化到 [0, 1]
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(eigenvector)
    _, U_std, _, _, _, _, _ = fuzzy.cluster.cmeans(
        data_normalized.T,  # 注意：数据需要是转置的格式，形状为 n_features x n_samples
        c=args.n_clusters,  # 簇的数量
        m=1.5,  # 模糊参数（通常设为 2）
        error=0.005,  # 终止条件的误差
        maxiter=1000,  # 最大迭代次数
        init=None,  # 初始隶属度矩阵（可以为空）
        seed=256  # 随机种子
    )
    print(f'FCM on data_normalized, the U matrix is:\n{U_std}')


    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    scale_U = min_max_scaler.fit_transform(U_std)
    print(f'the scale U:{scale_U}')




