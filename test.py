import numpy as np

# 示例数据，包含多个数字列表
data = np.array([
    [1, 2, 3, 4, 5],
    [2, 3, 72, 5, 6],
    [3, 4, 5, 100, 7]
], dtype=object)

# 将所有列表合并为一个一维数组
all_numbers = np.concatenate(data)

# 找出唯一的数字
unique_numbers = np.unique(all_numbers)

# 输出唯一数字及其数量
print("Unique numbers:", unique_numbers)
print("Number of unique numbers:", len(unique_numbers))
