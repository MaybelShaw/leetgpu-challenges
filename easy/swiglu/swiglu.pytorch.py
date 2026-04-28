import torch


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    halfN = N // 2
    x = input[:halfN]
    gate = input[halfN:N]
    output[:halfN].copy_(x * gate / (1.0 + torch.exp(-gate)))
