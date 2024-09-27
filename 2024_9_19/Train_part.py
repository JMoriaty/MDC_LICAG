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
from utils import multiViewDataset,cluster_acc
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


    dataset = multiViewDataset(args.dataset, args.viewNumber, pretrain=True)
    dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(model.parameters(), lr=args.lr)

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

    dataset = multiViewDataset(args.dataset, args.viewNumber, True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(model.parameters(), lr=args.lr)

    for batch_index, (x, y, idx) in enumerate(dataloader):
        for view_index in range(args.viewNumber):
            x[view_index] = x[view_index].to(device)
        output = model(x)

    loss_function = nn.KLDivLoss(reduction='mean')

    print("Start Self-supervised Learning！")

    #--------------- 正式训练epoch ----------------
    for epoch in tqdm.tqdm(range(1000)):
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

        # for view_index in range(args.viewNumber):


        view_specific_term = 1 * MSE_loss + args.gamma * KL_loss


        Loss = view_specific_term

        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

        # 每次epoch中acc记录
        # wandb.log({"loss": Loss,
        #            "MSE_loss": MSE_loss,
        #            "KL_loss": KL_loss,
        #            "acc": acc
        #            })


        # print(' MSE_loss:{:.4f}'.format(MSE_loss),
        #       ',KL_loss:{:.4f}'.format(KL_loss)
        #       )
        logging.info(f'\nEpoch [{epoch + 1}/{1000}]')
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
            # print('Aata in {:.f}-th view'.format(view_index), ': Acc {:.4f}'.format(acc_inView),
            #       ', nmi {:.4f}'.format(nmi_inView), ', ari {:.4f}'.format(ari_inView))
            # print(f"Data in {view_index}-th view :\t Acc:{acc_inView:.4f}\t NMI:{nmi_inView:.4f}\t ARI:{ari_inView:.4f}")
            logging.info(
                f'Data in {view_index}-th, Accuracy: {acc_inView:.4f}, MSE_loss: {MSE_loss:.4f}, KL_loss: {KL_loss:.4f}')

    # %%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%
