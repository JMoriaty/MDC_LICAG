# -*- coding: utf-8 -*-
# @Time    : 2024/7/21 15:46
# @Author  : Ginger
# @FileName: DMC_LICAG.py
# @Software: PyCharm

'''
将多视图IDEC和LICAG融合的实验1
'''
import os
from LICAG import *
import argparse
import numpy as np
import tqdm
from  sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import cluster_acc, WKLDiv, multiViewDataset2
import torch.nn.functional as F
import time

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

import warnings
warnings.filterwarnings('ignore')

import wandb
wandb.require("core")


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

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

    def LICAG_part(self, x):
        # #对原始data进行LICAG
        LICAG_loss = 0.
        _, H = LICAG(x, args.dimofA, args.n_anchors, args.n_neighbors)
        kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)
        kmeans.fit_predict(H)

        n, m = H.shape[0], kmeans.cluster_centers_.shape[0]
        for j in range(n):
            for i in range(m):
                A = torch.tensor(H[j])
                B = torch.tensor(kmeans.cluster_centers_[i])
                LICAG_loss += F.mse_loss(A, B)

        return LICAG_loss

    def forward(self, x, pretrain):
        outputs = []
        loss3 = 0.
        for viewIndex in range(self.viewNumber):
            outputs.append(self.AEs[viewIndex](x[viewIndex]))

        if not pretrain:
            x_cpu = [tensor.cpu().detach().numpy() for tensor in x]
            loss3 = self.LICAG_part(x_cpu)

        return outputs, loss3

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
        output, _ = model(x, pretrain=True)

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
        output,_ = model(x,pretrain=True)
        y = y.data.cpu().numpy()

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)

    for viewIndex in range(args.viewNumber):
        z_v = output[viewIndex][1]
        kmeans.fit_predict(z_v.cpu().detach().data.numpy())
        model.AEs[viewIndex].clusteringLayer.centroids.data = torch.tensor(kmeans.cluster_centers_).to(device)

    y_pred = kmeans.labels_
    acc = cluster_acc(y, y_pred)
    nmi = nmi_score(y, y_pred)
    ari = ari_score(y, y_pred)


    print('Acc {:.4f}'.format(acc),
          ', nmi {:.4f}'.format(nmi),
          ', ari {:.4f}'.format(ari))


    torch.save(model.state_dict(), args.save_path)


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

    dataset = multiViewDataset2(args.dataset, args.viewNumber, args.method, False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(model.parameters(), lr = args.lr)


    for batch_index, (x, y, idx) in enumerate(dataloader):
        for view_index in range(args.viewNumber):
            x[view_index] = x[view_index].to(device)
        output,_ = model(x, pretrain=False)


    loss_function = nn.KLDivLoss(reduction='mean')



    #对x进行LICAG
    # x_cpu = [tensor.cpu().detach().numpy() for tensor in x]
    # LICAG_loss = LICAG_part(x_cpu)

    print("Start Self-supervised Learning！")
    for epoch in tqdm.tqdm(range(1000)):
        qlist = list()
        plist = list()

        for batch_index, (x, y, idx) in enumerate(dataloader):
            y = y.data.cpu().numpy()
            MSE_loss = 0.
            # view_loss = 0.
            KL_loss = 0.
            LICAG_loss = 0.

            for view_index in range(args.viewNumber):
                x[view_index] = x[view_index].to(device)
            output, LICAG_loss = model(x, pretrain=False)


        arrays = []
        for view_index in range(args.viewNumber):
            MSE_loss += F.mse_loss(output[view_index][0], x[view_index])
            q_temp = output[view_index][2]      #shape is [2000, 10]
            p_temp = output[view_index][3]

            qlist.append(q_temp)
            plist.append(p_temp)

            arrays.append(q_temp.cpu().detach().numpy())



        for view_index in range(args.viewNumber):

            input = qlist[view_index].log()
            target = plist[view_index]
            KL_loss += loss_function(input, target)


        # #对z进LICAG
        # for view_index in range(args.viewNumber):
        #     z_assemble = [np.array(output[i][1].cpu().detach().numpy()) for i in range(view_index)]        #len(z_assemble) = 6 type is list
        #     # z_assemble = [np.array(output[i][1]) for i in range(view_index)]
        #
        # #这里z_assemble进去运算的时候是融合之后再进行的返回，而q确实单个视图未融合的
        # stacked_arrays = np.stack(arrays)
        # q_mean = np.mean(stacked_arrays, axis=0)
        # q_normalized = normalize(q_mean, axis=1, norm='l1')
        # LICAG_loss = LICAG_part(z_assemble, q_normalized)


        Loss = 1 * MSE_loss + args.gamma * KL_loss + LICAG_loss
        optimizer.zero_grad()
        Loss.backward(retain_graph=True)
        optimizer.step()

        for view_index in range(args.viewNumber):
            z_temp = output[view_index][1]
            if view_index == 0:
                z_all = z_temp
            else:
                z_all = torch.cat((z_all, z_temp), 1)
        kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)
        kmeans.fit_predict(z_all.cpu().detach().data.numpy())

        y_pred = kmeans.labels_
        acc = cluster_acc(y, y_pred)

        wandb.log({"learning_rate": args.lr,
                   "loss": Loss,
                   "MSE_loss": MSE_loss,
                   "KL_loss": KL_loss,
                   "LICAG_loss": LICAG_loss,
                   "acc": acc
                   })


        if epoch % 100 == 0:

            print(' acc:{:.4f}'.format(acc),
                    'MSE_loss:{:.4f}'.format(MSE_loss),
                  ',KL_loss:{:.4f}'.format(KL_loss),
                  ',LICAG_loss:{:.4}'.format(LICAG_loss))



    #对Z聚类
    for batch_index,(x, y, _)in enumerate(dataloader):
        y = y.data.cpu().numpy()
        for view_index in range(args.viewNumber):
            x[view_index] = x[view_index].to(device)
    output, LICAG_loss = model(x, pretrain=False)

    for view_index in range(args.viewNumber):
        z_temp = output[view_index][1]

        if view_index == 0:
            z_all = z_temp
        else:
            z_all = torch.cat((z_all, z_temp), 1)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)
    kmeans.fit_predict(z_all.cpu().detach().data.numpy())

    y_pred = kmeans.labels_
    acc = cluster_acc(y, y_pred)
    nmi = nmi_score(y, y_pred)
    ari = ari_score(y, y_pred)


    print('Acc {:.4f}'.format(acc),
          ', nmi {:.4f}'.format(nmi),
          ', ari {:.4f}'.format(ari))

def setup_seed(seed=100):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    setup_seed()
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=7, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--dataset', type=str, default='HW')
    parser.add_argument('--arch', type=int, default=50)
    parser.add_argument('--gamma', type=int, default=1)
    parser.add_argument('--method', type=str, default='HW')
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--dimofA', type=int, default=20)
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
    print(args)


    start =time.time()
    t0 = time.time()


    # wandb.init(project='DMC_LICAG', name=time.strftime('%y-%m-%d(%H:%M)'))
    wandb.init(project='DMC_LICAG', name=time.strftime('add LICAG to forward'))

    if not os.path.exists(args.save_path):
        Pre_Train_AEs()
        Training()
    else:
        print("已存在预训练pkl!")
        Training()

    t1 = time.time()
    print("Total time:",(t1 - t0))

