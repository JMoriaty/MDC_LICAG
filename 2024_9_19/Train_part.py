# -*- coding: utf-8 -*-
# @Time    : 2024/9/23 10:20
# @Author  : Ginger
# @FileName: Train_part.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
# @Time    : 2024/9/23 10:13
# @Author  : Ginger
# @FileName: Multi-view-IDEC.py
# @Software: PyCharm

from config import get_config
from net_part import MIDEC
import numpy as np
import torch.nn as nn
import tqdm
from  sklearn.cluster import KMeans
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import multiViewDataset,cluster_acc,multiViewDataset2,log_header
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import logging



import warnings
warnings.filterwarnings('ignore')

import wandb
wandb.require("core")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = get_config()

#%%%%%%%%%%%%%%% 预训练网络 %%%%%%%%%%%%%%%%
def Pre_Train():
    save_path = args.save_path
    viewNumber = args.viewNumber
    model = MIDEC(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=2000,
        n_dec_1=2000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        pretrain=True,
        save_path=save_path,
        viewNumber=viewNumber
    ).to(device)


    dataset = multiViewDataset2(args.dataset, args.viewNumber, pretrain=True)
    dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(model.parameters(), lr=args.lr)

    print("Start pretraining!")
    for epoch in tqdm.tqdm(range(args.epoch)):

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
        output = model(x)           #这里的output是preTrain=Ture下的返回值

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)

    for viewIndex in range(args.viewNumber):
        z_v = output[viewIndex][1]
        kmeans.fit_predict(z_v.cpu().detach().data.numpy())
        model.AEs[viewIndex].clusteringLayer.centroids.data = torch.tensor(kmeans.cluster_centers_).to(device)


    torch.save(model.state_dict(), args.save_path)
    print("Successful save pre-trained model")

#%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%




#%%%%%%%%%%%%%%%%%% 多视图IDEC训练 %%%%%%%%%%%%%%%%%%%%%
def Multi_view_IDEC():

    model = MIDEC(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=2000,
        n_dec_1=2000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        pretrain=False,
        save_path=args.save_path,
        viewNumber= args.viewNumber
    ).to(device)
    model.load_state_dict(torch.load(args.save_path))

    dataset = multiViewDataset2(args.dataset, args.viewNumber, True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(model.parameters(), lr=args.lr)

    for batch_index, (x, y, idx) in enumerate(dataloader):
        for view_index in range(args.viewNumber):
            x[view_index] = x[view_index].to(device)
        output = model(x)

    loss_function = nn.KLDivLoss(reduction='mean')

    print("Start Self-supervised Learning！")

    log_header(args.dataset)

    #--------------- 正式训练epoch ----------------
    for epoch in tqdm.tqdm(range(args.epoch)):
        qlist = list()
        plist = list()

        for batch_index, (x, y, idx) in enumerate(dataloader):
            MSE_loss = 0.
            KL_loss = 0.

            for view_index in range(args.viewNumber):
                x[view_index] = x[view_index].to(device)
            output = model(x)

        for view_index in range(args.viewNumber):
            MSE_loss += F.mse_loss(output[view_index][0], x[view_index])
            q_temp = output[view_index][2]
            p_temp = output[view_index][3]

            qlist.append(q_temp)
            plist.append(p_temp)

        for view_index in range(args.viewNumber):
            input = qlist[view_index].log()
            target = plist[view_index]
            KL_loss += loss_function(input, target)


        view_specific_term = 1 * MSE_loss + args.gamma * KL_loss

        Loss = view_specific_term

        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()



        if epoch % 250 == 0:
            print(' MSE_loss:{:.4f}'.format(MSE_loss),
                  ',KL_loss:{:.4f}'.format(KL_loss)
                  )

        #------------------------- log file writing -------------------
        if epoch % 50 == 0 or (epoch+1) == args.epoch:
            logging.info(f'\n Epoch [{epoch + 1}/{args.epoch}]')
            for view_index in range(args.viewNumber):

                z_temp = output[view_index][1]
                kmeans = KMeans(n_clusters=args.n_clusters, n_init=100)
                kmeans.fit_predict(z_temp.cpu().detach().data.numpy())
                y_pred_temp = kmeans.labels_

                # y_pred_temp = qlist[view_index].argmax(1)
                # y_pred_temp = y_pred_temp.cpu().numpy().astype(np.int64)

                acc_inView = cluster_acc(y, y_pred_temp)
                nmi_inView = nmi_score(y, y_pred_temp)
                ari_inView = ari_score(y, y_pred_temp)

                logging.info(
                    f'Data in {view_index}-th, Accuracy: {acc_inView:.4f}, nmi_inView:{nmi_inView:.4f},ari_inView:{ari_inView:.4f},MSE_loss: {MSE_loss:.4f}, KL_loss: {KL_loss:.4f}')

                # # ----------wandb每次epoch中acc记录----------
                # wandb.log({"loss": Loss,
                #            "MSE_loss": MSE_loss,
                #            "KL_loss": KL_loss,
                #            "acc": acc
                #            })
                # # ------------------- END ----------------

        # ---------------------------------- END -----------------------------------

    #---------------------对训练后的特征输出FCM--------------------
    import skfuzzy as fuzzy
    U_viewlist = []
    for view_index in range(args.viewNumber):
        z_temp = output[view_index][1]
        Z = z_temp.cpu().detach().data.numpy()

        _, U_inView, _, _, _, _, _ = fuzzy.cluster.cmeans(
            Z.T,  # 注意：数据需要是转置的格式，形状为 n_features x n_samples
            c=args.n_clusters,  # 簇的数量
            m=2,  # 模糊参数（通常设为 2）
            error=0.005,  # 终止条件的误差
            maxiter=1000,  # 最大迭代次数
            init=None,  # 初始隶属度矩阵（可以为空）
            seed=42  # 随机种子
        )

        U_viewlist.append(U_inView.T)

    return U_viewlist

    #-------------------------END-----------------------------


    # %%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%
