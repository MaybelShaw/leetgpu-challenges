import torch


# X, Y are tensors on the GPU
def solve(X: torch.Tensor, Y: torch.Tensor, N: int):
    Y[:N].copy_(1 / (1 + torch.exp(-X[:N])))
