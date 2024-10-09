# -*- coding: utf-8 -*-
# @Time    : 2024/9/19 16:30
# @Author  : Ginger
# @FileName: bone.py
# @Software: PyCharm


import os
import numpy as np
import torch
import time
from config import get_config
import logging
import torch.nn as nn


import wandb
wandb.require("core")


import warnings
warnings.filterwarnings('ignore')

from Extract_H_part import Extract_H
from Train_part import Pre_Train,Multi_view_IDEC
from torch.utils.data import DataLoader
from utils import multiViewDataset,cluster_acc,multiViewDataset2,log_header
from  sklearn.cluster import KMeans
import skfuzzy as fuzzy


#%%%%%%%%%%%%%%%%% 初始化参数 %%%%%%%%%%%%%%%%%%%%%
def Initialization():

    dataset = multiViewDataset2(args.dataset, args.viewNumber, True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)


    #初始化H矩阵
    for batch_index, (x, y, _) in enumerate(dataloader):
        labels = y
        X = x
    _, H = Extract_H(X, args.dimofH, args.n_anchors, args.n_neighbors)
    # H = torch.tensor(H).to(device)


    #初始化H的聚类中心O,shape is (类数,降维后H中元素的维度)
    kmeans.fit_predict(H)
    O_init = kmeans.cluster_centers_
    O_tensor = torch.tensor(O_init).to(device)
    O = O_tensor.float().requires_grad_(True)        # 将其转换为可以参与反向传播的 Tensor

    #初始化融合矩阵F
    _, F_temp, _, _, _, _, _ = fuzzy.cluster.cmeans(
            H.T,  # 注意：数据需要是转置的格式，形状为 n_features x n_samples
            c=args.n_clusters,  # 簇的数量
            m=2,  # 模糊参数（通常设为 2）
            error=0.005,  # 终止条件的误差
            maxiter=1000,  # 最大迭代次数
            init=None,  # 初始隶属度矩阵（可以为空）
            seed=42  # 随机种子
        )
    F_tensor = torch.tensor(F_temp.T).to(device)
    F = F_tensor.float().requires_grad_(True)

    #初始化指示矩阵Plist[]
    Plist = list()
    P_temp = np.eye(args.instanceNumber)
    P_temp_device = torch.tensor(P_temp).to(device)
    for view_index in range(args.viewNumber):
        Plist.append(P_temp_device)


#%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%



#%%%%%%%%%%%%%%%%% 正式训练 %%%%%%%%%%%%%%%%%%%%%



def Train():
    Multi_view_IDEC()

    # Fusion-term()

#%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%



if __name__ == '__main__':


    # #------------- wandb-log-init ------------------
    # wandb.init(project='DMC_LICAG_new2', name=time.strftime('%y-%m-%d(%H:%M)'))
    # #-------------- END ------------------------

    #随机种子
    np.random.seed(5489)

    #参数打印
    args = get_config()
    args_dict = vars(args)
    print(f"{'Parameter':<20} {'Value':<20}")
    print("=" * 40)

    for key, value in args_dict.items():
        if isinstance(value, list):
            # 将列表中的元素转换为字符串
            value = ', '.join(map(str, value))
        print(f"{key:<20} {value:<20}")
    print("=" * 40)


    # #-------------- log file init -----------------
    from datetime import datetime

    current_time = datetime.now().strftime("%Y-%m-%d_%H_%M")
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # 如果目录不存在，创建它

    # 使用时间戳生成日志文件名，并指定路径
    log_filename = os.path.join(log_dir, f'{current_time}_{args.dataset}.log')
    print(f"Log file will be saved to: {log_filename}")

    # 配置日志记录
    logging.basicConfig(
        filename=log_filename,  # 动态生成的日志文件名及路径
        level=logging.INFO,  # 日志级别
        format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
    )
    # #--------------------- END -----------------------


    #训练硬件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #运行时间模块
    start = time.time()
    t0 = time.time()

    #初始化参数
    Initialization()

    #是否进行预训练
    if not os.path.exists(args.save_path):
        Pre_Train()
        Train()
    else:
        print("已存在预训练pkl!")
        Train()

    #运行时间打印
    end = time.time()
    print("Total time:",(end - start))







