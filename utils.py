from __future__ import division, print_function
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, v_measure_score
import scipy.io as scio
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
import argparse

nmi = normalized_mutual_info_score
vmeasure = v_measure_score
ari = adjusted_rand_score

class multiViewDataset(Dataset):

    def __init__(self, dataName, viewNumber, pretrain):
        dataPath = './data/' + dataName + '.mat'
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

class multiViewDataset2(Dataset):

    def __init__(self, dataName, viewNumber, pretrain):
        dataPath = './data/' + dataName + '.mat'
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

class imagedataset(Dataset):

    def __init__(self, dataName,viewNumber,method,pretrain):
        dataPath = './data/' + dataName + '.mat'
        matData = scio.loadmat(dataPath)
        self.data=[]
        self.viewNumber = viewNumber
        for viewIndex in range(viewNumber):
            temp=matData['X'+str(viewIndex+1)].astype(np.float32)
            if self.viewNumber>=6:
                temp=min_max_scaler.fit_transform(temp)
            self.data.append(temp)
        Y = matData['Y'][0]
        self.labels = Y
        self.pretrain= not pretrain




    def __getitem__(self, index):
        data_tensor=[]
        for viewIndex in range(self.viewNumber):
            m=self.data[viewIndex][index]
            data_tensor.append(torch.from_numpy(self.data[viewIndex][index]))
        label= self.labels[index]
        label_tensor=torch.tensor(label)
        index_tensor = torch.tensor(index)

        return data_tensor, label_tensor, index_tensor


    def __len__(self):
        return len(self.labels)


class WKLDiv(torch.nn.Module):
    def __init__(self):
        super(WKLDiv, self).__init__()

    def forward(self, q_logit, p, w):
        p_logit=torch.log(p + 1e-12)
        kl = torch.sum(p * (p_logit- q_logit)*w, 1)
        return torch.mean(kl)


#######################################################
# Evaluate Critiron
#######################################################

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


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
    parser.add_argument('--dimofH', type=int, default=10)
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

    dataset = multiViewDataset2(args.dataset, args.viewNumber, args.method, pretrain=True)
    dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # for batch_idx, (data_tensor, label_tensor, index_tensor) in enumerate(dataset):
    #     print(f"Batch {batch_idx + 1}:")
    #     print(f"Data (from multiple views): {data_tensor}")  # 列表，包含来自不同视图的张量
    #     print(f"Label: {label_tensor}")  # 样本标签张量
    #     print(f"Index: {index_tensor}")  # 样本索引张量
