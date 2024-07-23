# -*- coding: utf-8 -*-
# @Time    : 2024/7/23 10:41
# @Author  : Ginger
# @FileName: LICAG_GPU.py
# @Software: PyCharm


import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import random
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def LICAG(X, dim, n_anchors, n_neighbors):
    n_views = len(X)
    n_samples = X[0].shape[0]

    if dim == 0:
        dim = n_anchors

    if n_neighbors == 0 or n_neighbors > n_anchors - 1:
        n_neighbors = n_anchors - 1

    X_assemble = torch.hstack([torch.tensor(x, device=device) for x in X])

    random.seed(5489)

    if n_samples > 10000:
        tmpIdx = random.sample(range(n_samples), 10000)
        subset = X_assemble[tmpIdx, :]
    else:
        subset = X_assemble

    kmeans = KMeans(n_clusters=n_anchors, max_iter=20, n_init=5, random_state=5489).fit(subset.cpu().numpy())
    anchors = torch.tensor(kmeans.cluster_centers_, device=device)

    W = torch.zeros((n_samples + n_views * n_anchors, n_samples + n_views * n_anchors), device=device)
    beg = 0
    for viewIndex in range(n_views):
        x_viewIndex = torch.tensor(X[viewIndex], device=device)
        n_features = x_viewIndex.shape[1]
        anchor_viewIndex = anchors[:, beg:beg + n_features]

        Z = anchor_graph_construction(x_viewIndex, anchor_viewIndex, n_neighbors)

        W[n_samples + (viewIndex) * n_anchors: n_samples + (viewIndex + 1) * n_anchors, 0: n_samples] = Z
        W[0: n_samples, n_samples + (viewIndex) * n_anchors: n_samples + (viewIndex + 1) * n_anchors] = Z.T

        beg += n_features

    normalized = True
    L = laplacian(W, normalized)
    eigenvalue, eigenvactor = torch.linalg.eigh(L)
    eigenvalue = eigenvalue[:n_samples].T
    return eigenvalue, eigenvactor[:n_samples, :]


def anchor_graph_construction(data, anchors, m):
    n = data.shape[0]
    n_anchors = anchors.shape[0]

    E = torch.cdist(anchors, data, p=2) ** 2

    SortedE, _ = torch.sort(E, dim=0)
    Eim1 = SortedE[m, :].repeat(n_anchors, 1)

    IndMask = torch.zeros((n_anchors, n), device=device)
    Ind = torch.argsort(E, dim=0)
    for i in range(n):
        IndMask[Ind[:m, i], i] = 1

    E_numerator = E * IndMask
    Eim1_numerator = Eim1 * IndMask

    denominator = m * Eim1 - torch.sum(E_numerator, dim=0) + torch.finfo(torch.float32).eps
    Z = (Eim1_numerator - E_numerator) / denominator

    return Z


def laplacian(W, normalized):
    n = W.shape[0]
    d = torch.sum(W, dim=1)
    if normalized:
        Dn = torch.diag(d ** -0.5)
        L = torch.eye(n, device=device) - Dn @ W @ Dn
    else:
        D = torch.diag(d)
        L = D - W
    return L


if __name__ == '__main__':
    torch.manual_seed(100)
    X = [torch.rand(100, 5, device=device), torch.rand(100, 10, device=device)]
    dim = 6
    n_anchors = 10
    n_neighbors = 5

    print("the type of x[0]", type(X[0]))

    eigenvalue, eigenvactor = LICAG(X, dim, n_anchors, n_neighbors)
    print("the shape of eigenvalue", eigenvalue.shape)
    print("the shape of eigenvactor", eigenvactor.shape)

    from time import time
    start =time()
    t0 = time()
    t1 = time()
    print("Total time:",(t1 - t0))