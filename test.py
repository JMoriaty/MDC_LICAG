import torch

# 生成6个形状为 (2000, 10) 的随机张量
tensors = [torch.rand(2000, 10) for _ in range(6)]

# 将这些张量放入一个列表中
X = tensors

# 打印每个张量的形状，以验证
for i, tensor in enumerate(X):
    print(f"Tensor {i+1} shape: {tensor.shape}")
    print("x[i]",X[0].shape)
