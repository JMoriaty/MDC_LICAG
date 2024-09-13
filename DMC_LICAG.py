# -*- coding: utf-8 -*-
# @Time    : 2024/7/21 15:46
# @Author  : Ginger
# @FileName: DMC_LICAG.py
# @Software: PyCharm

'''
将多视图IDEC和LICAG融合的实验1
'''
import os

import numpy

from LICAG import *
import argparse
import numpy as np
import tqdm
from  sklearn.cluster import KMeans
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import cluster_acc, WKLDiv, multiViewDataset2
import torch.nn.functional as F
import skfuzzy as fuzzy
import time

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

import wandb
wandb.require("core")

import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"



#------------------IDEC聚类项-----------------------
class ClusteringLayer(nn.Module):       #输入Z^v，计算聚类中心\mu，计算q分布并返回
    def __init__(self, n_clusters, n_z):
        super(ClusteringLayer, self).__init__()
        self.centroids = Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True)

    def forward(self, x):
        #torch.sum对输入的tensor数据的某一维度求和
        q = 1.0 / (1 + torch.sum(torch.pow(x.unsqueeze(1) - self.centroids, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()

        p = q ** 2 / q.sum(0)
        p = (p.t() / p.sum(1)).t()

        return q, p
#---------------------------------------------------------



#------------------LICAG聚类项-----------------------------
class ClusteringLayer_Latent(nn.Module):
    def __int__(self,n_clusters, H):
        super(ClusteringLayer_Latent, self).__init__()
        self.centroids = Parameter(torch.Tensor(n_clusters, H), requires_grad=True)

    def forward(self, x):
        kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)
        kmeans.fit_predict(x)
        O = kmeans.cluster_centers_

        return O

#------------------------------------------------------------

class SingleViewModel(nn.Module):
    def __init__(
            self,
            n_enc_1,
            n_enc_2,
            n_enc_3,
            n_dec_1,
            n_dec_2,
            n_dec_3,
            n_input,
            n_z,
            pretrain
    ):
        super(SingleViewModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_enc_1),
            nn.ReLU(inplace=True),
            nn.Linear(n_enc_1, n_enc_2),
            nn.ReLU(inplace=True),
            nn.Linear(n_enc_2, n_enc_3),
            nn.ReLU(inplace=True),
            nn.Linear(n_enc_3, n_z)
        )
        for m in self.encoder:  # 遍历self.encoder中的每一个模块
            if isinstance(m, torch.nn.Linear):  # 检查模块是否是torch.nn.Linear类型
                torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))  # 使用Xavier均匀分布初始化权重
                torch.nn.init.constant_(m.bias, 0)  # 将偏置初始化为零

        self.decoder = nn.Sequential(
            nn.Linear(n_z, n_dec_1),
            nn.ReLU(inplace=True),
            nn.Linear(n_dec_1, n_dec_2),
            nn.ReLU(inplace=True),
            nn.Linear(n_dec_2, n_dec_3),
            nn.ReLU(inplace=True),
            nn.Linear(n_dec_3, n_input)
        )

        for m in self.decoder:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                torch.nn.init.constant_(m.bias, 0)


        self.pretrain = pretrain
        self.clusteringLayer = ClusteringLayer(args.n_clusters, n_z)

    def forward(self, x):
        z = self.encoder(x)

        if self.pretrain:
            x_rebuilt = self.decoder(z)
            return x_rebuilt, z

        x_rebuilt = self.decoder(z)

        q,p = self.clusteringLayer(z)

        return x_rebuilt, z, q, p


class MultiViewModel(nn.Module):
    def __init__(
        self,
        n_enc_1,
        n_enc_2,
        n_enc_3,
        n_dec_1,
        n_dec_2,
        n_dec_3,
        n_input,
        n_z,
        n_clusters,
        pretrain,
        save_path
    ):
        super(MultiViewModel, self).__init__()
        self.pretrain = pretrain
        self.save_path = save_path
        self.n_clusters = n_clusters
        self.viewNumber = args.viewNumber

        AEs = []

        for viewIndex in range(self.viewNumber):
            AEs.append(SingleViewModel(
                n_enc_1=n_enc_1,
                n_enc_2=n_enc_2,
                n_enc_3=n_enc_3,
                n_dec_1=n_dec_1,
                n_dec_2=n_dec_2,
                n_dec_3=n_dec_3,
                n_input=n_input[viewIndex],
                n_z=n_z,
                pretrain=self.pretrain
            ))

        self.AEs = nn.ModuleList(AEs)

    def forward(self, x):
        outputs = []
        for viewIndex in range(self.viewNumber):
            outputs.append(self.AEs[viewIndex](x[viewIndex]))

        return outputs

def Pre_Train_AEs():
    save_path = args.save_path
    viewNumber = args.viewNumber
    model = MultiViewModel(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=2000,
        n_dec_1=2000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.arch,
        pretrain=True,
        save_path=save_path
    ).to(device)

    dataset = multiViewDataset2(args.dataset, args.viewNumber, args.method, pretrain=True)
    dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(model.parameters(), lr = args.lr)

    print("Start pretraining!")
    for epoch in tqdm.tqdm(range(1000)):
        for batch_index, (x, _, _) in enumerate(dataLoader):
            loss = 0.0
            for viewIndex in range(viewNumber):
                x[viewIndex] = x[viewIndex].to(device)
        output = model(x)

        for viewIndex in range(viewNumber):
            loss += F.mse_loss(output[viewIndex][0], x[viewIndex])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print("\n mseloss loss", loss)

    for batch_index, (x, y, _) in enumerate(dataLoader):
        for viewIndex in range(args.viewNumber):
            x[viewIndex] = x[viewIndex].to(device)
        output = model(x)
        y = y.data.cpu().numpy()

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)

    for viewIndex in range(args.viewNumber):
        z_v = output[viewIndex][1]
        kmeans.fit_predict(z_v.cpu().detach().data.numpy())
        model.AEs[viewIndex].clusteringLayer.centroids.data = torch.tensor(kmeans.cluster_centers_).to(device)      #这行代码理解一下


    torch.save(model.state_dict(), args.save_path)
    print("Successful save pre-trained model")

def LICAG_part(x):
    # #对原始data进行LICAG
    LICAG_loss = 0.
    _, H = LICAG(x, args.dimofH, args.n_anchors, args.n_neighbors)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)
    kmeans.fit_predict(H)

    return H

def Training():
    model = MultiViewModel(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=2000,
        n_dec_1=2000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.arch,
        pretrain=False,
        save_path=args.save_path
    ).to(device)
    model.load_state_dict(torch.load(args.save_path))

    dataset = multiViewDataset2(args.dataset, args.viewNumber, args.method, True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(model.parameters(), lr = args.lr)


    for batch_index, (x, y, idx) in enumerate(dataloader):
        for view_index in range(args.viewNumber):
            x[view_index] = x[view_index].to(device)
        output = model(x)

        # for view_index in range(args.viewNumber):
        #     z[view_index] = output[view_index][1]

    loss_function = nn.KLDivLoss(reduction='mean')

    #---------------跨视图潜在特征H计算----------------------
    x_cpu = [tensor.cpu().detach().numpy() for tensor in x]
    H = LICAG_part(x_cpu)
        #F融合矩阵初始化
    _, F_temp, _, _, _, _, _ = fuzzy.cluster.cmeans(
        H.T,  # 注意：数据需要是转置的格式，形状为 n_features x n_samples
        c=args.n_clusters,  # 簇的数量
        m=2,  # 模糊参数（通常设为 2）
        error=0.005,  # 终止条件的误差
        maxiter=1000,  # 最大迭代次数
        init=None,  # 初始隶属度矩阵（可以为空）
        seed=42  # 随机种子
    )
    Fusion = torch.from_numpy(F_temp.T).float()     #F先转置，在由numpy转换为tensor,同时将double转换为float
        #F的聚类中心O初始化
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)
    kmeans.fit_predict(H)
    O = kmeans.cluster_centers_

    #-----------------------------------------------------

    print("Start Self-supervised Learning！")
    for epoch in tqdm.tqdm(range(200)):
        qlist = list()
        plist = list()
        P = list()

        for batch_index, (x, y, idx) in enumerate(dataloader):
            MSE_loss = 0.
            KL_loss = 0.
            Consensus_learning_term = 0.

            for view_index in range(args.viewNumber):
                x[view_index] = x[view_index].to(device)
            output = model(x)

        for view_index in range(args.viewNumber):
            MSE_loss += F.mse_loss(output[view_index][0], x[view_index])
            q_temp = output[view_index][2]      #shape is [2000, 10]
            p_temp = output[view_index][3]

            qlist.append(q_temp)
            plist.append(p_temp)
            P.append((np.eye(args.instanceNumber)))

        for view_index in range(args.viewNumber):

            input = qlist[view_index].log()
            target = plist[view_index]
            KL_loss += loss_function(input, target)

            # diff = Fusion - numpy.dot(P[view_index], qlist[view_index].cpu().detach().data.numpy()
            P_tensor = torch.from_numpy(P[view_index]).float()
            diff = Fusion.to(device) - torch.mm(P_tensor.to(device), qlist[view_index])

            Consensus_learning_term += (np.linalg.norm(diff.cpu().detach().data.numpy(), 'fro') ** 2)

        view_specific_term = 1 * MSE_loss + args.gamma * KL_loss

        Distance_HO = cdist(H, O, metric='euclidean') ** 2  # H与O之间的距离矩阵，n*m。
        Latent_information_guidance = np.multiply(Distance_HO, Fusion).to(device)

        Consensus_learning_term = torch.tensor(Consensus_learning_term)
        Consensus_learning_term =  Consensus_learning_term.float()
        Consensus_learning_term = Consensus_learning_term.to(device)

        Loss = view_specific_term + args.beta * Latent_information_guidance + Consensus_learning_term

        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

        # #----------------------每次epoch中acc记录---------------------------
        # for batch_index, (x, y, _) in enumerate(dataloader):
        #     y = y.data.cpu().numpy()
        #     for view_index in range(args.viewNumber):
        #         x[view_index] = x[view_index].to(device)
        # output = model(x)
        #
        # for view_index in range(args.viewNumber):
        #     z_temp = output[view_index][1]
        #     if view_index == 0:
        #         z_all = z_temp
        #     else:
        #         z_all = torch.cat((z_all, z_temp), 1)
        #
        # kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)
        # kmeans.fit_predict(z_all.cpu().detach().data.numpy())
        # y_pred = kmeans.labels_
        # acc = cluster_acc(y, y_pred)
        # wandb.log({"loss": Loss,
        #            "MSE_loss": MSE_loss,
        #            "KL_loss": KL_loss,
        #            "acc": acc
        #            })
        # #---------------------------------------------------------


        if epoch % 20 == 0:
            print(' MSE_loss:{:.4f}'.format(MSE_loss),
                  ',KL_loss:{:.4f}'.format(KL_loss)
                  )

    #---------------------各视图低维特征z的提取与拼接--------------------
    for batch_index,(x, y, _)in enumerate(dataloader):
        y = y.data.cpu().numpy()
        for view_index in range(args.viewNumber):
            x[view_index] = x[view_index].to(device)
    output = model(x)

    for view_index in range(args.viewNumber):
        z_temp = output[view_index][1]
        if view_index == 0:
            z_all = z_temp
        else:
            z_all = torch.cat((z_all, z_temp), 1)
    #-----------------------------------------------------------------

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)
    kmeans.fit_predict(z_all.cpu().detach().data.numpy())

    y_pred = kmeans.labels_
    acc = cluster_acc(y, y_pred)
    nmi = nmi_score(y, y_pred)
    ari = ari_score(y, y_pred)


    print('Acc {:.4f}'.format(acc),
          ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

    # #-------------对H进行测试---------------
    # print("the shape of H",H.shape)
    # kmeans.fit_predict(H)
    # y_pred_madebyH =kmeans.labels_
    # acc_H = cluster_acc(y, y_pred_madebyH)
    # print('Acc made by H: {:.4f}'.format(acc_H))
    # print('y_pred_madebyH',y_pred_madebyH)
    ##------------------------------------------




def setup_seed(seed=100):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=7, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--dataset', type=str, default='HW')
    parser.add_argument('--arch', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--beta', type=float, default=10)
    parser.add_argument('--method', type=str, default='HW') #好像没用？
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--dimofH', type=int, default=20)
    parser.add_argument('--n_anchors', type=int,default=50)
    parser.add_argument('--n_neighbors',type=int, default=10)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'HW':
        args.n_input = [216, 76, 64, 6, 240, 47]
        args.viewNumber = 6
        args.instanceNumber = 2000
        args.batch_size = 2000
        args.n_clusters = 10
        args.save_path = './data/HW.pkl'
        args.arch = 50
        args.gamma = 0.1

    if args.dataset == 'WebKB':
        args.n_input = [1840,3000]
        args.viewNumber = 2
        args.instanceNumber = 1051
        args.batch_size = 1051
        args.n_clusters = 6
        args.save_path = './data/WebKB.pkl'
        args.arch = 50
        args.gamma = 0.1

    start = time.time()
    t0 = time.time()

    print(args)

    #--------------wandb-log----------------------
    # wandb.init(project='DMC_LICAG_new1', name=time.strftime('%y-%m-%d(%H:%M)'))
    # wandb.init(project='DMC_LICAG', name=time.strftime('add LICAG(z) to forward'))
    #---------------------------------------------

    if not os.path.exists(args.save_path):
        Pre_Train_AEs()
        Training()
    else:
        print("已存在预训练pkl!")
        Training()

    t1 = time.time()
    print("Total time:",(t1 - t0))

