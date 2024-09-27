# -*- coding: utf-8 -*-
# @Time    : 2024/9/19 10:42
# @Author  : Ginger
# @FileName: showdata.py
# @Software: PyCharm

import os
import scipy.io
import numpy as np


# 加载 .mat 文件的通用函数
def load_and_inspect_mat_file(file_path):
    print(f"\n\n---------------Loading file: {file_path}--------------------------")

    # 使用 scipy.io.loadmat 加载文件
    mat_data = scipy.io.loadmat(file_path)

    # 显示 .mat 文件中的所有变量名（跳过以 '__' 开头的元数据）
    variables = [var for var in mat_data.keys() if not var.startswith('__')]
    print(f"Variables in the file: {variables}")

    # 遍历每个变量并显示其维度和数据类型
    for var in variables:
        data = mat_data[var]
        if isinstance(data, np.ndarray):
            print(f"\nVariable Name: {var}")
            print(f"Shape: {data.shape}")
            print(f"Data Type: {data.dtype}")


            # 如果数据很大，先只显示部分内容
            if data.size <= 10:  # 如果数据很小，显示完整数据
                print(f"Data: {data}")
            else:
                print(f"First 5 elements of {var}: {data.flat[:5]}")

            if var == 'X':
                for index in range(data.shape[0]):
                    for i, array in enumerate(data[index]):
                        print(f"Shape of data[{index}] is: {array.shape}")

            if var == 'y':
                all_numbers = np.concatenate(data)
                unique_numbers = np.unique(all_numbers)
                print("Number of clusters:", len(unique_numbers))


        else:
            print(f"{var} is not an ndarray.")


# 自动化加载多个文件的函数
def process_multiple_datasets(directory):
    # 列出指定目录下的所有 .mat 文件
    mat_files = [f for f in os.listdir(directory) if f.endswith('.mat')]

    # 如果目录下没有 .mat 文件
    if not mat_files:
        print("No .mat files found in the directory.")
        return

    # 依次处理每个 .mat 文件
    for mat_file in mat_files:
        file_path = os.path.join(directory, mat_file)
        load_and_inspect_mat_file(file_path)


# 设置包含数据集的文件夹路径
directory = './dataset'  # 替换为包含 .mat 文件的文件夹路径

# 自动处理该目录下的所有 .mat 文件
process_multiple_datasets(directory)
