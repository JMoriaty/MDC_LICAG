# -*- coding: utf-8 -*-
# @Time    : 2024/9/20 9:41
# @Author  : Ginger
# @FileName: utils.py
# @Software: PyCharm

from __future__ import division, print_function
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, v_measure_score
import scipy.io as scio
from torch.utils.data import DataLoader
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
from config import get_config

class multiViewDataset(Dataset):

    def __init__(self, dataName, viewNumber, pretrain):
        dataPath = './dataset/' + dataName + '.mat'
        matData = scio.loadmat(dataPath)
        self.data = []
        self.viewNumber = viewNumber

        #----------- 对各个视图的X进行水平上拼接 -------------
        for viewIndex in range(viewNumber):                 #viewIndex指定某个视图
            dataX = matData['X'][viewIndex]                 #dataX指该视图中的X数据
            # temp = dataX[0]
            temp = min_max_scaler.fit_transform(dataX[0])
            # print(f"this is data in view{[viewIndex]}:{temp}")
            self.data.append(temp)

        # print("self.data type",type(self.data))
        # print(self.data)
        #--------------------- END ----------------------



        #---------- 提取每个元素的标签，并拼接成一行(有元素个数列) ----------------
        dataY = matData['y']
        Y = []
        for index in range(dataY.shape[0]):
            for i, array in enumerate(dataY[index]):
                temp = array
            Y.append(temp)
        self.labels = Y
        #---------------------- END----------------------------

        self.pretrain = not pretrain

    def __getitem__(self, index):
        data_tensor = []
        for viewIndex in range(self.viewNumber):
            m = self.data[viewIndex]
            # data_tensor.append(torch.from_numpy(self.data[viewIndex][index]))
            data_tensor.append(torch.from_numpy(self.data[viewIndex][index].astype(np.float32)))
        label = self.labels[index]
        label_tensor = torch.tensor(label, dtype=torch.float32)
        index_tensor = torch.tensor(index)

        return data_tensor, label_tensor, index_tensor


    def __len__(self):
        return len(self.labels)


def cluster_acc(y_true, y_pred):
    y_true = y_true.cpu().numpy().astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


# def cluster_acc(y_true, y_pred):
#     # 将 PyTorch 张量转换为 NumPy 数组并确保数据类型一致
#     y_true = y_true.cpu().numpy().astype(np.int64)
#     y_pred = y_pred.cpu().numpy().astype(np.int64)
#
#     # 确保 y_pred 和 y_true 的大小相同
#     assert y_pred.size == y_true.size
#
#     # 计算最大类别数 + 1
#     D = max(y_pred.max(), y_true.max()) + 1
#
#     # 初始化匹配矩阵
#     w = np.zeros((D, D), dtype=np.int64)
#
#     # 填充匹配矩阵
#     for i in range(y_pred.size):
#         w[y_pred[i], y_true[i]] += 1
#
#     # 线性分配问题的解决
#     row_ind, col_ind = linear_sum_assignment(w.max() - w)
#
#     # 计算匹配后的准确率
#     acc = sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size
#
#     return acc





if __name__ == '__main__':

    args = get_config()

    dataset = multiViewDataset(args.dataset, args.viewNumber, pretrain=True)
    dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    for batch_idx, (data_tensor, label_tensor, index_tensor) in enumerate(dataLoader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Data (from multiple views): {data_tensor}")  # 列表，包含来自不同视图的张量
        print(f"Label: {label_tensor}")  # 样本标签张量
        print(f"Index: {index_tensor}")  # 样本索引张量
        print(batch_idx)

