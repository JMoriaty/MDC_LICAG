# -*- coding: utf-8 -*-
# @Time    : 2024/9/19 15:40
# @Author  : Ginger
# @FileName: net_part.py
# @Software: PyCharm

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
from  sklearn.cluster import KMeans

#%%%%%%%%%%%%%%%%%%%%%%%% 聚类层 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class ClusteringLayer(nn.Module):       #输入Z^v，计算聚类中心\mu，计算q分布并返回
    def __init__(self, n_clusters, n_z):
        super(ClusteringLayer, self).__init__()
        self.centroids = Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True)       #\mu.IDEC当中对z的聚类中心


    def forward(self, x):

        #torch.sum对输入的tensor数据的某一维度求和
        q = 1.0 / (1 + torch.sum(torch.pow(x.unsqueeze(1) - self.centroids, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()

        p = q ** 2 / q.sum(0)
        p = (p.t() / p.sum(1)).t()

        return q, p
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%% 单视图IDEC %%%%%%%%%%%%%%%%%%%%%%%%%%%%
class IDEC(nn.Module):
    def __init__(
            self,
            n_enc_1,
            n_enc_2,
            n_enc_3,
            n_z,
            n_dec_1,
            n_dec_2,
            n_dec_3,
            n_input,
            pretrain,
            n_clusters
    ):
        super(IDEC, self).__init__()


        #----------- 构建解码3层线性网络 -----------
        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_enc_1),
            nn.ReLU(inplace=True),
            nn.Linear(n_enc_1, n_enc_2),
            nn.ReLU(inplace=True),
            nn.Linear(n_enc_2, n_enc_3),
            nn.ReLU(inplace=True),
            nn.Linear(n_enc_3, n_z)
        )
        #---------------- END ---------------------


        #----------- 初始化编码器权重(w)和偏执值(bias) --------
        for m in self.encoder:  # 遍历self.encoder中的每一个模块
            if isinstance(m, torch.nn.Linear):  # 检查模块是否是torch.nn.Linear类型
                torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))  # 使用Xavier均匀分布初始化权重
                torch.nn.init.constant_(m.bias, 0)  # 将偏置初始化为零
        #------------------------ END ---------------------


        #----------- 构建解码3层线性网络 ------------
        self.decoder = nn.Sequential(
            nn.Linear(n_z, n_dec_1),
            nn.ReLU(inplace=True),
            nn.Linear(n_dec_1, n_dec_2),
            nn.ReLU(inplace=True),
            nn.Linear(n_dec_2, n_dec_3),
            nn.ReLU(inplace=True),
            nn.Linear(n_dec_3, n_input)
        )
        #-------------- END -------------------



        # ------------ 初始化解码器权重(w)和偏执值(bias) --------
        for m in self.decoder:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                torch.nn.init.constant_(m.bias, 0)
        #----------------------- END ------------------------


        self.pretrain = pretrain
        self.clusteringLayer = ClusteringLayer(n_clusters, n_z)


    #--------------- 前向传播 -----------------
    def forward(self, x):
        z = self.encoder(x)

        if self.pretrain:
            x_rebuilt = self.decoder(z)
            return x_rebuilt, z

        x_rebuilt = self.decoder(z)

        q,p = self.clusteringLayer(z)


        return x_rebuilt, z, q, p
    #-------------------- END ----------------

#%%%%%%%%%%%%%%%%%%%%%%%%%%% 单视图IDEC(END) %%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%% 多视图IDEC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class MIDEC(nn.Module):
    def __init__(
        self,
        n_enc_1,
        n_enc_2,
        n_enc_3,
        n_z,
        n_dec_1,
        n_dec_2,
        n_dec_3,
        n_input,
        viewNumber,
        n_clusters,
        pretrain,
        save_path
    ):
        super(MIDEC, self).__init__()
        self.pretrain = pretrain
        self.save_path = save_path
        self.n_clusters = n_clusters
        self.viewNumber = viewNumber

        AEs = []

        #------- 构建每个视图的IDEC ------------
        for viewIndex in range(self.viewNumber):
            AEs.append(IDEC(
                n_enc_1=n_enc_1,
                n_enc_2=n_enc_2,
                n_enc_3=n_enc_3,
                n_z=n_z,
                n_dec_1=n_dec_1,
                n_dec_2=n_dec_2,
                n_dec_3=n_dec_3,
                n_input=n_input[viewIndex],
                pretrain=self.pretrain,
                n_clusters=n_clusters
            ))
        #--------------END----------------

        self.AEs = nn.ModuleList(AEs)   #保存模型

    #------------ 多视图网络的前向传播 -----------
    def forward(self, x):
        outputs = []
        for viewIndex in range(self.viewNumber):
            outputs.append(self.AEs[viewIndex](x[viewIndex]))

        return outputs
    #----------------- END -----------------

#%%%%%%%%%%%%%%%%%%%%%%% 多视图IDEC(END) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%