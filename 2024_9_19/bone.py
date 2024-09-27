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


import wandb
wandb.require("core")


import warnings
warnings.filterwarnings('ignore')

from Train_part import Pre_Train,Multi_view_IDEC



#%%%%%%%%%%%%%%%%% 正式训练 %%%%%%%%%%%%%%%%%%%%%

def Train():
    Multi_view_IDEC()
    # Latent-H()
    # Fusion-term()

#%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%



if __name__ == '__main__':

    # wandb-log
    # wandb.init(project='DMC_LICAG_new2', name=time.strftime('%y-%m-%d(%H:%M)'))
    logging.basicConfig(
        filename='training.log',  # 日志文件名
        level=logging.INFO,  # 日志级别
        format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
    )


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


    #训练硬件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #运行时间模块
    start = time.time()
    t0 = time.time()

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







