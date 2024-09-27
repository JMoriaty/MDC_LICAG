import numpy as np

# 假设 darry 是一个包含多个 ndarray 的 NumPy 数组
darry = np.array([np.random.rand(3, 3), np.random.rand(3, 3), np.random.rand(3, 3)], dtype=object)

# 将 darry 转换为 Python 列表
list_of_darrays = list(darry)

# 输出结果
print("转换后的列表:")
for i, arr in enumerate(list_of_darrays):
    print(f"视角 {i+1}: \n{arr}\n")
