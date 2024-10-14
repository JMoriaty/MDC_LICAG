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
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class multiViewDataset2(Dataset):

    def __init__(self, dataName, viewNumber, pretrain):
        dataPath = './dataset/' + dataName + '.mat'
        matData = scio.loadmat(dataPath)
        self.data = []
        self.viewNumber = viewNumber
        for viewIndex in range(viewNumber):
            temp = matData['X'+str(viewIndex+1)].astype(np.float32)
            if self.viewNumber >= 2:
                temp = min_max_scaler.fit_transform(temp)

            self.data.append(temp)
        Y = matData['Y'][0]
        self.labels = Y
        self.pretrain = not pretrain


    def __getitem__(self, index):
        data_tensor = []
        for viewIndex in range(self.viewNumber):
            m = self.data[viewIndex][index]
            data_tensor.append(torch.from_numpy(self.data[viewIndex][index]))
        label = self.labels[index]
        label_tensor = torch.tensor(label)
        index_tensor = torch.tensor(index)

        return data_tensor, label_tensor, index_tensor


    def __len__(self):
        return len(self.labels)

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


def compute_sum_of_distances_matrix(H, O, F):
    # 假设 H: (n, d), O: (m, d), F: (n, m)

    if isinstance(H, np.ndarray):
        H = torch.tensor(H)
    if isinstance(O, np.ndarray):
        O = torch.tensor(O)
    if isinstance(F, np.ndarray):
        F = torch.tensor(F).to(device)

    # 扩展 H 和 O，计算每个 h_j 和 o_i 之间的 L2 范数的平方
    H_expanded = H.unsqueeze(0).to(device)  # (1, n, d)
    O_expanded = O.unsqueeze(1)  # (m, 1, d)

    # 计算 ||h_j - o_i||^2
    distance_squared = torch.sum((H_expanded - O_expanded) ** 2, dim=2)  # (m, n)

    # 按元素乘以 F，然后求和
    total_sum = torch.sum(distance_squared.T * F)

    return total_sum


def log_header(dataset_name):
    # 定义表格内容
    border = '+' + '-' * 30 + '+'
    dataset_line = f"| Dataset: {dataset_name:<20} |"

    # 打印表格内容到日志中
    logging.info(border)
    logging.info(dataset_line)
    logging.info(border)


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

