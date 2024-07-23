import numpy as np

# 假设你的数组存储在output中，每个output[viewIndex][2]是一个形状为(200, 10)的数组
output = {
    1: {2: np.random.rand(200, 10)},
    2: {2: np.random.rand(200, 10)},
    3: {2: np.random.rand(200, 10)},
    4: {2: np.random.rand(200, 10)},
    5: {2: np.random.rand(200, 10)},
    6: {2: np.random.rand(200, 10)}
}

# 使用列表存储所有viewIndex从1到6的数组
arrays = [output[viewIndex][2] for viewIndex in range(1, 7)]

# 将所有数组沿新轴堆叠
stacked_arrays = np.stack(arrays)

# 求和并求平均
mean_array = np.mean(stacked_arrays, axis=0)

# 打印最终形状以检查
print("Mean shape:", mean_array.shape)  # 应该是 (200, 10)

print(mean_array)
