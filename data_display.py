# -*- coding: utf-8 -*-
# @Time    : 2024/7/21 13:05
# @Author  : Ginger
# @FileName: data_display.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import numpy as np

import argparse
from torch.utils.data import DataLoader
from utils import cluster_acc, WKLDiv, multiViewDataset2


def display_dataset_info(X, y):
    """
    展示数据集的具体信息。

    参数：
    X (numpy.ndarray or pandas.DataFrame): 特征矩阵。
    y (numpy.ndarray or pandas.Series): 目标标签。

    返回：
    None
    """
    if isinstance(X, pd.DataFrame):
        print("数据集信息：")
        print(X.info())
        print("\n数据集的统计摘要：")
        print(X.describe())
    else:
        print(f"数据集大小：{X.shape}")
        print(f"特征数量：{X.shape[1]}")

    unique_classes, counts_classes = np.unique(y, return_counts=True)
    print("\n目标标签分布情况：")
    for cls, count in zip(unique_classes, counts_classes):
        print(f"标签 {cls}：{count} 个样本")

    print("\n前5条数据记录：")
    if isinstance(X, pd.DataFrame):
        print(X.head())
    else:
        print(pd.DataFrame(X).head())

    print("\n前5个目标标签：")
    print(y[:5])

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=7, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--dataset', type=str, default='HW')
    parser.add_argument('--arch', type=int, default=50)
    parser.add_argument('--gamma', type=int, default=1)
    parser.add_argument('--method', type=str, default='HW')
    parser.add_argument('--epoch', type=int, default=1000)
    args = parser.parse_args()

    if args.dataset == 'HW':
        args.n_input = [216, 76, 64, 6, 240, 47]
        args.viewNumber = 6
        args.instanceNumber = 2000
        args.batch_size = 2000
        args.n_clusters = 10
        args.save_path = './data/HW.pkl'
        args.arch = 50
        args.gamma = 0.1


    # 示例使用
    # 假设我们有一个特征矩阵 X 和目标标签 y
    dataset = multiViewDataset2(args.dataset, args.viewNumber, args.method, pretrain=True)
    dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # X, y = data.data, data.target
    for batch_index, (x, y, _) in enumerate(dataLoader):
        for viewIndex in range(args.viewNumber):
            x[viewIndex] = x[viewIndex]
        y = y.data.cpu().numpy()
        display_dataset_info(x[viewIndex], y)
        print("len of x", len(x))
        print(x)
        print("type of x", type(x))




