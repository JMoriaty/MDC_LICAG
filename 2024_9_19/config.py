# -*- coding: utf-8 -*-
# @Time    : 2024/9/19 17:38
# @Author  : Ginger
# @FileName: config.py
# @Software: PyCharm

# config.py
import argparse
import torch

def get_config():
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--dataset', type=str, default='NUS-WIDE')
    parser.add_argument('--viewNumber', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--beta', type=float, default=10)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--dimofH', type=int, default=10)
    parser.add_argument('--n_anchors', type=int, default=50)
    parser.add_argument('--n_neighbors',type=int, default=10)
    parser.add_argument('--data_dir',type=str, default='./dataset/')


    args = parser.parse_args()

    if args.dataset == 'MNIST-10k':
        args.n_input = [30, 9, 30]
        args.viewNumber = 3
        args.instanceNumber = 10000
        args.batch_size = 10000
        args.n_clusters = 10
        args.save_path = './dataset/MNIST-10k.pkl'
        args.gamma = 0.1

    elif args.dataset == 'Movies':
        args.n_input = [1878, 1398]
        args.viewNumber = 2
        args.instanceNumber = 617
        args.batch_size = 617
        args.n_clusters = 17
        args.save_path = './dataset/Movies.pkl'
        args.gamma = 0.1

    elif args.dataset == 'NUS-WIDE':
        args.n_input = [64,144,73,128,225]
        args.viewNumber = 5
        args.instanceNumber = 2400
        args.batch_size = 2400
        args.n_clusters = 12
        args.save_path = './dataset/NUS-WIDE.pkl'
        args.gamma = 0.1

    elif args.dataset == 'Reuters-1500':
        args.n_input = [2153, 24893, 34279, 15506, 11547]
        args.viewNumber = 5
        args.instanceNumber = 1500
        args.batch_size = 1500
        args.n_clusters = 6
        args.save_path = './dataset/Reuters-1500.pkl'
        args.gamma = 0.1

    elif args.dataset == 'WebKB':
        args.n_input = [1840, 3000]
        args.viewNumber = 2
        args.instanceNumber = 1051
        args.batch_size = 1051
        args.n_clusters = 6
        args.save_path = './dataset/WebKB.pkl'
        args.gamma = 0.1

    elif args.dataset == 'Wikipedia':
        args.n_input = [128, 10]
        args.viewNumber = 2
        args.instanceNumber = 2866
        args.batch_size = 2866
        args.n_clusters = 10
        args.save_path = './dataset/Wikipedia.pkl'
        args.gamma = 0.1

    return args
